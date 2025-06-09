from histocartography.ml import HACTModel
from histocartography.ml.layers.constants import GNN_NODE_FEAT_IN, GNN_NODE_FEAT_OUT
from typing import Dict, Union
import torch
import dgl


class CustomHACTNet(HACTModel):
    """
    Custom HACT model that extends the base HACTModel to include additional functionality or modifications.
    """

    def __init__(self, return_cell_graph_embedding:bool, **kwargs):
        """
        Custom HACT model constructor.

        Args:
            cg_gnn_params (Dict): Cell Graph GNN configuration parameters.
            tg_gnn_params (Dict): Tissue Graph GNN configuration parameters.
            classification_params (Dict): Classification configuration parameters.
            cg_node_dim (int): Cell node feature dimension.
            tg_node_dim (int): Tissue node feature dimension.
        """
        super().__init__(**kwargs)
        self.return_cell_graph_embedding = return_cell_graph_embedding

    def forward(
            self,
            cell_graph: Union[dgl.DGLGraph, dgl.batch],
            tissue_graph: Union[dgl.DGLGraph, dgl.batch],
            assignment_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Foward pass.

        Args:
            cell_graph (Union[dgl.DGLGraph, dgl.batch]): Cell graph or Batch of cell graphs.
            tissue_graph (Union[dgl.DGLGraph, dgl.batch]): Tissue graph or Batch of tissue graphs.
            assignment_matrix (torch.Tensor): List of assignment matrices

        Returns:
            torch.Tensor: model output.
        """

        # 1. GNN layers over the low level graph
        ll_feats = cell_graph.ndata[GNN_NODE_FEAT_IN]
        ll_h = self.cell_graph_gnn(cell_graph, ll_feats, with_readout=False)

        # 2. Sum the low level features according to assignment matrix
        ll_h_concat = self._compute_assigned_feats(
            cell_graph, ll_h, assignment_matrix)

        tissue_graph.ndata[GNN_NODE_FEAT_IN] = torch.cat(
            (ll_h_concat, tissue_graph.ndata[GNN_NODE_FEAT_IN]), dim=1)

        # 3. GNN layers over the high level graph
        hl_feats = tissue_graph.ndata[GNN_NODE_FEAT_IN]
        graph_embeddings = self.superpx_gnn(tissue_graph, hl_feats)

        # 4. Classification layers
        logits = self.pred_layer(graph_embeddings)

        if self.return_cell_graph_embedding:
            return logits, graph_embeddings

        return logits
