#!/usr/bin/env python3
"""
Script for training CG-GNN, TG-GNN and HACT models
"""

import argparse
import os
import shutil
import uuid
from typing import Any

import pytorch_lightning as pl
import mlflow
import mlflow.pytorch
import pandas as pd
import torch
import yaml
from dataloader import make_data_loader
from histocartography.ml import CellGraphModel, HACTModel, TissueGraphModel
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm
from torch.nn.functional import cross_entropy
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.rich_model_summary import RichModelSummary
from pytorch_lightning.callbacks.progress import RichProgressBar

from torchmetrics import F1Score, Accuracy

class HistoCartographyModel(pl.LightningModule):
    """
    Base class for HistoCartography models.
    """
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model

        self.train_f1 = F1Score(task="multiclass", average="weighted", num_classes=3)
        self.train_detailed_f1 = F1Score(task="multiclass", average=None, num_classes=3)

        self.val_f1 = F1Score(task="multiclass", average="weighted", num_classes=3)
        self.val_detailed_f1 = F1Score(task="multiclass", average=None, num_classes=3)

        self.test_f1 = F1Score(task="multiclass", average="weighted", num_classes=3)
        self.test_detailed_f1 = F1Score(task="multiclass", average=None, num_classes=3)

        self.save_hyperparameters(kwargs)

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        self.train_detailed_f1.reset()

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()

        f1_0, f1_1, f1_2 = self.val_detailed_f1.compute()
        self.log_dict(
            {
                "val_f1_0": f1_0,
                "val_f1_1": f1_1,
                "val_f1_2": f1_2,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.hparams.batch_size,
        )

        self.val_detailed_f1.reset()

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()

        f1_0, f1_1, f1_2 = self.test_detailed_f1.compute()
        self.log_dict(
            {
                "test_f1_0": f1_0,
                "test_f1_1": f1_1,
                "test_f1_2": f1_2,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.hparams.batch_size,
        )

        self.test_detailed_f1.reset()



    def forward(self, *args):
        return self.model(*args)

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.
        """
        data = batch[:-1]
        labels = batch[-1]
        logits = self.forward(*data)
        loss = cross_entropy(logits, labels)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.batch_size)

        self.train_f1(logits, labels)
        f1_0, f1_1, f1_2 = self.train_detailed_f1(logits, labels)
        self.log_dict(
            {
                "train_weighted_f1_score": self.train_f1,
                "train_f1_0": f1_0,
                "train_f1_1": f1_1,
                "train_f1_2": f1_2,
            },
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.hparams.batch_size,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.
        """
        data = batch[:-1]
        labels = batch[-1]
        logits = self.forward(*data)
        loss = cross_entropy(logits, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.batch_size)

        self.val_f1(logits, labels)
        self.val_detailed_f1.update(logits, labels)
        self.log("val_weighted_f1_score", self.val_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.batch_size)

        return {"val_loss": loss, "logits": logits, "labels": labels}

    def test_step(self, batch, batch_idx):
        """
        Test step for the model.
        """
        data = batch[:-1]
        labels = batch[-1]
        logits = self.forward(*data)
        loss = cross_entropy(logits, labels)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.batch_size)

        self.test_f1(logits, labels)
        self.test_detailed_f1.update(logits, labels)
        self.log("test_weighted_f1_score", self.test_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.hparams.batch_size)

        return {"test_loss": loss, "logits": logits, "labels": labels}


    def configure_optimizers(self) -> Any:
        """
        Configure the optimizer for the model.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, weight_decay=5e-4)
        return optimizer




# cuda support
IS_CUDA = torch.cuda.is_available()
DEVICE = "cuda:0" if IS_CUDA else "cpu"
NODE_DIM = 514


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cg_path", type=str, help="path to the cell graphs.", default=None, required=False)
    parser.add_argument("--tg_path", type=str, help="path to tissue graphs.", default=None, required=False)
    parser.add_argument(
        "--assign_mat_path", type=str, help="path to the assignment matrices.", default=None, required=False
    )
    parser.add_argument(
        "-conf", "--config_fpath", type=str, help="path to the config file.", default="", required=False
    )
    parser.add_argument("--model_path", type=str, help="path to where the model is saved.", default="", required=False)
    parser.add_argument(
        "--in_ram",
        help="if the data should be stored in RAM.",
        action="store_true",
    )
    parser.add_argument("-b", "--batch_size", type=int, help="batch size.", default=1, required=False)
    parser.add_argument("--epochs", type=int, help="epochs.", default=10, required=False)
    parser.add_argument("-l", "--learning_rate", type=float, help="learning rate.", default=10e-3, required=False)
    parser.add_argument(
        "--out_path",
        type=str,
        help="path to where the output data are saved (currently only for the interpretability).",
        default="../../data/graphs",
        required=False,
    )
    parser.add_argument(
        "--logger", type=str, help='Logger type. Options are "mlflow" or "none"', required=False, default="none"
    )

    return parser.parse_args()


def main(args):
    """
    Train HACTNet, CG-GNN or TG-GNN.

    Args:
        args (Namespace): parsed arguments.

    """
    # load config file
    with open(args.config_fpath) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # log parameters to logger
    if args.logger == "mlflow":
        mlflow.log_params({"batch_size": args.batch_size})
        df = pd.io.json.json_normalize(config)
        rep = {"graph_building.": "", "model_params.": "", "gnn_params.": ""}  # replacement for shorter key names
        for i, j in rep.items():
            df.columns = df.columns.str.replace(i, j)
        flatten_config = df.to_dict(orient="records")[0]
        for key, val in flatten_config.items():
            mlflow.log_params({key: str(val)})

    # make data loaders (train, validation & test)
    train_dataloader = make_data_loader(
        cg_path=os.path.join(args.cg_path, "train") if args.cg_path is not None else None,
        tg_path=os.path.join(args.tg_path, "train") if args.tg_path is not None else None,
        assign_mat_path=os.path.join(args.assign_mat_path, "train") if args.assign_mat_path is not None else None,
        batch_size=args.batch_size,
        load_in_ram=args.in_ram,
    )
    val_dataloader = make_data_loader(
        cg_path=os.path.join(args.cg_path, "val") if args.cg_path is not None else None,
        tg_path=os.path.join(args.tg_path, "val") if args.tg_path is not None else None,
        assign_mat_path=os.path.join(args.assign_mat_path, "val") if args.assign_mat_path is not None else None,
        batch_size=args.batch_size,
        load_in_ram=args.in_ram,
        shuffle=False,  # no need to shuffle validation data
    )
    test_dataloader = make_data_loader(
        cg_path=os.path.join(args.cg_path, "test") if args.cg_path is not None else None,
        tg_path=os.path.join(args.tg_path, "test") if args.tg_path is not None else None,
        assign_mat_path=os.path.join(args.assign_mat_path, "test") if args.assign_mat_path is not None else None,
        batch_size=args.batch_size,
        load_in_ram=args.in_ram,
        shuffle=False,  # no need to shuffle test data
    )

    # declare model
    if "bracs_cggnn" in args.config_fpath:
        model = CellGraphModel(
            gnn_params=config["gnn_params"],
            classification_params=config["classification_params"],
            node_dim=NODE_DIM,
            num_classes=3,
        ).to(DEVICE)

    elif "bracs_tggnn" in args.config_fpath:
        model = TissueGraphModel(
            gnn_params=config["gnn_params"],
            classification_params=config["classification_params"],
            node_dim=NODE_DIM,
            num_classes=3,
        ).to(DEVICE)

    elif "bracs_hact" in args.config_fpath:
        model = HACTModel(
            cg_gnn_params=config["cg_gnn_params"],
            tg_gnn_params=config["tg_gnn_params"],
            classification_params=config["classification_params"],
            cg_node_dim=NODE_DIM,
            tg_node_dim=NODE_DIM,
            num_classes=3,
        ).to(DEVICE)
    else:
        raise ValueError("Model type not recognized. Options are: TG, CG or HACT.")


    lightning_model = HistoCartographyModel(model=model, learning_rate=args.learning_rate, batch_size=args.batch_size)
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5, mode="min", min_delta=0.001),
            ModelCheckpoint(
                filename="best_model_loss",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            ),
            RichProgressBar(leave=True),
            RichModelSummary(),
        ],
    )

    # train the model
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    trainer.test(
        ckpt_path="best",
        dataloaders=test_dataloader,
    )


if __name__ == "__main__":
    main(args=parse_arguments())
