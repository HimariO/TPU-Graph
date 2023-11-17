import torch
from torch_geometric.graphgym.models.layer import SAGEConv, new_layer_config
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network
from torch_geometric.utils import to_dense_batch
import torch_geometric as pyg

from graphgps.layer.gatedgcn_layer import GatedGCNLayer
from graphgps.layer.gine_conv_layer import GINEConvLayer
from performer_pytorch import SelfAttention


class PerformerWrapper(torch.nn.Module):

    def __init__(self, conv_layer, dim_h, num_heads, dropout) -> None:
        super().__init__()
        self.conv_layer = conv_layer
        self.attn = SelfAttention(
            dim=dim_h, 
            heads=num_heads,
            dropout=dropout,
            causal=False
        )
        self.norm1 = torch.nn.BatchNorm1d(dim_h)
        self.norm2 = torch.nn.BatchNorm1d(dim_h)
        self.dropout = torch.nn.Dropout(p=cfg.gnn.dropout)
    
    def forward(self, batch):
        h = batch.x
        batch.x = self.norm1(batch.x)
        batch.x = self.dropout(batch.x)

        batch = self.conv_layer(batch)
        h_dense, mask = to_dense_batch(h, batch.batch)
        h_attn = self.attn(h_dense, mask=mask)[mask]
        
        batch.x = self.norm2(batch.x + h_attn + h)
        return batch


class SAGEConvLayer(torch.nn.Module):

    def __init__(self, layer_config, **kwargs):
        super().__init__()
        self.model = pyg.nn.SAGEConv(layer_config.dim_in, layer_config.dim_out,
                                     bias=layer_config.has_bias)
        # self.norm = torch.nn.BatchNorm1d(layer_config.dim_in)
        self.drop_prob = cfg.gnn.dropout
        self.residual = cfg.gnn.residual
        self.dropout = torch.nn.Dropout(p=cfg.gnn.dropout)

    def forward(self, batch):
        x_in = batch.x
        # batch.x = self.norm(batch.x)
        if self.drop_prob > 0:
            batch.x = self.dropout(batch.x)
        batch.x = self.model(batch.x, batch.edge_index)
        if self.residual:
            batch.x = batch.x + x_in
        return batch


@register_network('custom_tpu_gnn')
class CustomTpuGNN(torch.nn.Module):
    """
    GNN model that customizes the torch_geometric.graphgym.models.gnn.GNN
    to support specific handling of new conv layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        dim_in = 128
        self.encoder = FeatureEncoder(dim_in)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                cfg.gnn.dim_feat_enc, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."
        
        layer_type = cfg.gnn.layer_type.split('+')
        conv_type = layer_type[0]
        conv_model = self.build_conv_model(conv_type)
        
        layers = []
        layer_cfg = new_layer_config(dim_in, dim_in, 1, has_act=True, has_bias=True, cfg=cfg)
        for _ in range(cfg.gnn.layers_mp):
            layers.append(conv_model(layer_cfg))
            if len(layer_type) > 1 and layer_type[1] == 'performer':
                layers[-1] = PerformerWrapper(layers[-1], dim_in, 4, 0.2)
        self.gnn_layers = torch.nn.Sequential(*layers)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)

    def build_conv_model(self, model_type):
        if model_type == 'gatedgcnconv':
            return GatedGCNLayer
        elif model_type == 'gineconv':
            return GINEConvLayer
        elif model_type == 'sageconv':
            return SAGEConv
        elif model_type == 'sageconvlayer':
            return SAGEConvLayer
        else:
            raise ValueError("Model {} unavailable".format(model_type))

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch
