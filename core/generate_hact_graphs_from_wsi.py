"""
Extract HACT graphs for all the sample in the given dataset.
"""

import argparse

# Remove warning level logs from dgl and numpy
import logging
import os

import dgl
import h5py
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from dgl.data.utils import save_graphs
from histocartography.preprocessing import (
    AssignmnentMatrixBuilder,  # assignment matrix
    ColorMergedSuperpixelExtractor,  # tissue detector
    DeepFeatureExtractor,  # feature extractor
    KNNGraphBuilder,  # kNN graph builder
    MacenkoStainNormalizer,  # stain normalizer
    NucleiExtractor,  # nuclei detector
    RAGGraphBuilder,  # build graph
)
from openslide import OpenSlide
from tqdm import tqdm

logging.getLogger("dgl").setLevel(logging.WARNING)
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

MIN_NR_PIXELS = 50000
MAX_NR_PIXELS = 50000000
STAIN_NORM_TARGET_IMAGE = "../data/target.png"  # define stain normalization target image.

from hovernet import HoVerNet

TUMOR_TYPE_TO_LABEL = {
    "C": 2,
    "B": 1,
    "A": 0,
}


class CustomNucleiExtractor(NucleiExtractor):
    """
    Fixes the model loading for the NucleiExtractor.
    """

    def _load_model_from_path(self, model_path):
        state_dict = torch.load(model_path, map_location="cpu")
        model = HoVerNet(mode="fast", nr_types=6)
        model.load_state_dict(state_dict["desc"])

        self.model = model


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slide_path", type=str, help="path to the WSI data. Assuming they are .tiff files", required=True
    )
    parser.add_argument(
        "--save_path", type=str, help="path to save the cell graphs.", default="../data/", required=False
    )
    parser.add_argument(
        "--h5_path",
        type=str,
        help="Path to h5 files that contains coords to extract and their labels. "
        "Assuming they have same name as slides",
        required=True,
    )

    parser.add_argument(
        "--slide_csv", type=str, help="Path to CSV file with slide names and their labels.", required=True
    )
    return parser.parse_args()


class HACTBuilding:
    def __init__(self):
        # 1. define stain normalizer
        self.normalizer = MacenkoStainNormalizer(target_path=STAIN_NORM_TARGET_IMAGE)

        # 2. define CG builders
        self._build_cg_builders()

        # 3. define TG builders
        self._build_tg_builders()

        # 4. define assignment matrix builder
        self.assignment_matrix_builder = AssignmnentMatrixBuilder()

        # 5. define var to store image IDs that failed (for whatever reason)
        self.image_ids_failing = []

    def _build_cg_builders(self):
        # a define nuclei extractor
        self.nuclei_detector = CustomNucleiExtractor(model_path="../hovernet_fast_pannuke_type_tf2pytorch.tar")

        # b define feature extractor: Extract patches of 72x72 pixels around each
        # nucleus centroid, then resize to 224 to match ResNet input size.
        self.nuclei_feature_extractor = DeepFeatureExtractor(architecture="resnet34", patch_size=40, resize_size=224)

        # c define k-NN graph builder with k=5 and thresholding edges longer
        # than 50 pixels. Add image size-normalized centroids to the node features.
        # For e.g., resulting node features are 512 features from ResNet34 + 2
        # normalized centroid features.
        self.knn_graph_builder = KNNGraphBuilder(k=5, thresh=75, add_loc_feats=True)

    def _build_tg_builders(self):
        # a define nuclei extractor
        self.tissue_detector = ColorMergedSuperpixelExtractor(
            superpixel_size=500, compactness=20, blur_kernel_size=1, threshold=0.05, downsampling_factor=4
        )

        # b define feature extractor: Extract patches of 144x144 pixels all over
        # the tissue regions. Each patch is resized to 224 to match ResNet input size.
        self.tissue_feature_extractor = DeepFeatureExtractor(architecture="resnet34", patch_size=144, resize_size=224)

        # c define RAG builder. Append normalized centroid to the node features.
        self.rag_graph_builder = RAGGraphBuilder(add_loc_feats=True)

    def _save_nuclei_map(self, nuclei_centroids: np.ndarray, image: np.ndarray, graph: dgl.DGLGraph, image_name: str):
        # Assuming code is running inside "core" directory so we will create the directory relative to that
        os.makedirs("../nuclei_maps", exist_ok=True)

        nx_graph = graph.to_networkx()
        positions = {i: (nuclei_centroids[i, 0], nuclei_centroids[i, 1]) for i in range(len(nuclei_centroids))}

        plt.imshow(image, cmap="gray")
        nx.draw(
            nx_graph, node_size=2, edge_color="cyan", node_color="blue", with_labels=False, pos=positions, arrowsize=1
        )
        plt.title("Nuclei Map for {}".format(image_name.replace(".jpg", "")))
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join("../nuclei_maps", image_name.replace(".jpg", ".png")), dpi=200)

        plt.close()

    def _build_cg(self, image: np.ndarray, image_name: str):
        nuclei_map, nuclei_centroids = self.nuclei_detector.process(image)

        if len(nuclei_centroids) <= 3:
            print(f"Warning: {len(nuclei_centroids)} nuclei centroids are too few.")
            raise ValueError("Too few nuclei centroids.")

        features = self.nuclei_feature_extractor.process(image, nuclei_map)
        graph = self.knn_graph_builder.process(nuclei_map, features)
        # self._save_nuclei_map(nuclei_centroids, image, graph, image_name)

        return graph, nuclei_centroids

    def _build_tg(self, image: np.ndarray):
        superpixels, _ = self.tissue_detector.process(image)
        features = self.tissue_feature_extractor.process(image, superpixels)
        graph = self.rag_graph_builder.process(superpixels, features)
        return graph, superpixels

    def process(self, wsi_path: str, h5_path: str, wsi_info_path: str, save_path: str, split: str):
        # filename and ips columns
        slide_info = pd.read_csv(wsi_info_path)

        # 1. get image path
        slide_names = slide_info["filename"]
        labels = slide_info["ips"]
        total_images = 0

        print(f"*** Start analysing {len(slide_info)} slides ***")

        for label, slide_name in tqdm(zip(labels, slide_names), total=len(slide_names)):
            h5_filename = slide_name.replace(".tiff", ".h5")
            h5_file_path = os.path.join(h5_path, h5_filename)

            slide = OpenSlide(os.path.join(wsi_path, slide_name))
            h5 = h5py.File(h5_file_path, "r")
            coords = h5["coords"]
            total_images += len(coords)

            for i in range(len(coords)):
                image = slide.read_region(coords[i], 0, (1024, 1024)).convert("RGB")
                image = np.array(image)
                image_name = f"{os.path.basename(slide_name).replace('.tiff', '')}_{coords[i]}.jpg"
                nr_pixels = image.shape[0] * image.shape[1]
                image_label = TUMOR_TYPE_TO_LABEL[label]

                cg_out = os.path.join(save_path, "cell_graphs", split, image_name.replace(".jpg", ".bin"))
                tg_out = os.path.join(save_path, "tissue_graphs", split, image_name.replace(".jpg", ".bin"))
                assign_out = os.path.join(save_path, "assignment_matrices", split, image_name.replace(".jpg", ".h5"))

                # if file was not already created + not too big + not too small, then process
                if not self._exists(cg_out, tg_out, assign_out) and self._valid_image(nr_pixels):
                    # b. stain norm the image
                    try:
                        image = self.normalizer.process(image)
                    except:
                        print(f"Warning: {image_name} failed during stain normalization.")
                        self.image_ids_failing.append(image_name)

                    try:
                        cell_graph, nuclei_centroid = self._build_cg(image, image_name)
                        save_graphs(filename=cg_out, g_list=[cell_graph], labels={"label": torch.tensor([image_label])})
                    except Exception:
                        print(f"Warning: {image_name} failed during cell graph generation.")
                        # print(e)
                        # raise e
                        self.image_ids_failing.append(image_name)
                        continue

                    try:
                        tissue_graph, tissue_map = self._build_tg(image)
                        save_graphs(
                            filename=tg_out, g_list=[tissue_graph], labels={"label": torch.tensor([image_label])}
                        )
                    except:
                        print(f"Warning: {image_name} failed during tissue graph generation.")
                        self.image_ids_failing.append(image_name)
                        continue

                    try:
                        assignment_matrix = self.assignment_matrix_builder.process(nuclei_centroid, tissue_map)
                        with h5py.File(assign_out, "w") as output_file:
                            output_file.create_dataset(
                                "assignment_matrix",
                                data=assignment_matrix,
                                compression="gzip",
                                compression_opts=9,
                            )
                    except Exception as e:
                        print(f"Warning: {image_name} failed during assignment matrix generation.")
                        self.image_ids_failing.append(image_name)
                        raise e

                else:
                    print("Image:", image_name, " was already processed or is too large/small.")

        print(
            f"Out of {total_images} images, {total_images - len(self.image_ids_failing)} successful HACT graph generations."
        )
        print("Failing IDs are:", self.image_ids_failing)

    def _valid_image(self, nr_pixels):
        if nr_pixels > MIN_NR_PIXELS and nr_pixels < MAX_NR_PIXELS:
            return True
        return False

    def _exists(self, cg_out, tg_out, assign_out):
        if os.path.isfile(cg_out) and os.path.isfile(tg_out) and os.path.isfile(assign_out):
            return True
        return False


if __name__ == "__main__":
    # 1. handle i/o
    args = parse_arguments()
    if not os.path.isdir(args.image_path) or not os.listdir(args.image_path):
        raise ValueError("Data directory is either empty or does not exist.")

    split = ""
    if "train" in args.image_path:
        split = "train"
    elif "val" in args.image_path:
        split = "val"
    elif "test" in args.image_path:
        split = "test"
    else:
        split = "all"

    os.makedirs(os.path.join(args.save_path, "cell_graphs", split), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "tissue_graphs", split), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "assignment_matrices", split), exist_ok=True)

    # 2. generate HACT graphs one-by-one, will automatically
    # run on GPU if available.
    hact_builder = HACTBuilding()
    hact_builder.process(args.slide_path, args.h5_path, args.slide_csv, args.save_path, split)
