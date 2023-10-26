import copy
from contextlib import nullcontext
from typing import *

import torch
import numpy as np
import torch_geometric.nn as tnn
from torch import nn, Tensor
from torch_sparse import SparseTensor
from torch_geometric.data import Data, Batch
from torch_geometric.graphgym.config import cfg

from graphgps.history import History


def preprocess_batch(batch, model, num_sample_configs):
    
    # batch_list = batch.to_data_list()
    batch_list = batch
    processed_batch_list = []
    sample_idx = []
    for g in batch_list:
        sample_idx.append(torch.randint(0, g.num_config.item(), (num_sample_configs,)))
        g.y = g.y[sample_idx[-1]]
        g.config_feats = g.config_feats.view(g.num_config, g.num_config_idx, -1)[sample_idx[-1], ...]
        g.config_feats = g.config_feats.transpose(0,1)
        g.config_feats_full = torch.zeros(
            [
                g.num_nodes,
                num_sample_configs,
                g.config_feats.shape[-1]
            ], 
            device=g.config_feats.device
        )
        g.config_feats_full -= 1
        g.config_feats_full[g.config_idx, ...] = g.config_feats
        g.adj = SparseTensor(row=g.edge_index[0], col=g.edge_index[1], sparse_sizes=(g.num_nodes, g.num_nodes))
        processed_batch_list.append(g)
    return Batch.from_data_list(processed_batch_list), sample_idx


def batch_sample_graph_segs(batch: Union[Batch, List[Data]], num_sample_config=32, train=True, all_segment=False):
    # HACK: doing the reduant `to_data_list` here so every tensor in Data will be at least 1D
    batch_list = batch.to_data_list() if isinstance(batch, Batch) else batch
    batch_train_list = []
    batch_num_parts = []
    segments_to_train = []
    
    for i in range(len(batch_list)):
        partptr = batch_list[i].partptr.cpu().numpy()
        num_parts = len(partptr) - 1
        if train:
            batch_num_parts.extend([num_parts] * num_sample_config)
        else:
            batch_num_parts.append(num_parts)  # HACK
        segment_to_train = np.random.randint(num_parts)
        segments_to_train.append(segment_to_train)
        
        for j in range(num_parts):
            if j == segment_to_train or all_segment:
                start = int(partptr[j])
                length = int(partptr[j + 1]) - start

                N, E = batch_list[i].num_nodes, batch_list[i].num_edges
                data = copy.copy(batch_list[i])
                del data.num_nodes
                adj, data.adj = data.adj, None  # adj is a SparseTensor

                adj = adj.narrow(0, start, length).narrow(1, start, length)
                edge_idx = adj.storage.value()

                for key, item in data:
                    if isinstance(item, torch.Tensor) and item.size(0) == N:
                        # get subset of node features
                        data[key] = item.narrow(0, start, length)
                    elif isinstance(item, torch.Tensor) and item.size(0) == E and item.ndim > 1:
                        # get subset of edge features
                        data[key] = item[edge_idx]
                    else:
                        data[key] = item

                row, col, _ = adj.coo()
                data.edge_index = torch.stack([row, col], dim=0)
                # create same graph-segment for each layout config
                for k in range(len(data.y)):
                    unfold_g = Data(
                        edge_index=data.edge_index,
                        op_feats=data.op_feats,
                        op_code=data.op_code,
                        config_feats=data.config_feats_full[:, k, :], 
                        num_nodes=length,
                    )

                    for key in data.keys:
                        if key not in unfold_g.keys:
                            setattr(unfold_g, key, getattr(data, key))
                    for feat_key in cfg.dataset.extra_cfg_feat_keys:
                        sampled = getattr(data, feat_key)[:, k, :]
                        setattr(unfold_g, feat_key, sampled)
                    batch_train_list.append(unfold_g)

    return (
        batch,
        batch_list,
        batch_train_list,
        batch_num_parts,
        segments_to_train,
    )


def batch_sample_full(batch: Union[Batch, List[Data]], num_sample_config=32, train=True, all_segment=False):
    # HACK: doing the reduant `to_data_list` here so every tensor in Data will be at least 1D
    batch_list = batch.to_data_list() if isinstance(batch, Batch) else batch
    batch_num_parts = []
    segments_to_train = []
    batch_train_list = []
    
    for i in range(len(batch_list)):
        partptr = batch_list[i].partptr.cpu().numpy()
        num_parts = len(partptr) - 1
        if train:
            batch_num_parts.extend([1] * num_sample_config)
        else:
            batch_num_parts.append(1)  # HACK
        segments_to_train.append(0)
        
        data = batch_list[i]

        for k in range(len(data.y)):
            unfold_g = Data(
                edge_index=data.edge_index,
                op_feats=data.op_feats,
                op_code=data.op_code,
                config_feats=data.config_feats_full[:, k, :], 
                num_nodes=data.num_nodes,
            )

            for key in data.keys:
                if key not in unfold_g.keys:
                    setattr(unfold_g, key, getattr(data, key))
            for feat_key in cfg.dataset.extra_cfg_feat_keys:
                sampled = getattr(data, feat_key)[:, k, :]
                setattr(unfold_g, feat_key, sampled)
            batch_train_list.append(unfold_g)

    return (
        batch,
        batch_list,
        batch_train_list,
        batch_num_parts,
        segments_to_train,
    )


def cached_node_embed(
        batch_list: List[Data],
        sampled_idx,
        segments_to_train: List[int],
        emb_table: History
    ) -> List[torch.Tensor]:
    batch_other = []
    
    for i, data in enumerate(batch_list):
        partptr = data.partptr.cpu().numpy()
        num_parts = len(partptr) - 1
        
        if emb_table.embedding_dim == 1:
            for j in range(num_parts):
                if j == segments_to_train[i]:
                    continue
                for k in range(len(data.y)):
                    batch_other.append(
                        emb_table.pull(
                            data.partition_idx.cpu() + num_parts * sampled_idx[i][k] + j
                        )
                    )
        else:
            for k in range(len(data.y)):
                batch_other.append(
                    emb_table.pull(
                        data.graph_config_idx.cpu() + sampled_idx[i][k]
                    )
                )
    return batch_other



class TPUModel(torch.nn.Module):
    """
    Wrapper to handle feature embedding/encoding
    """
    def __init__(
            self, 
            model,
            input_feat_key=None,
            enc_config=False,
            enc_tile_config=False,
            extra_cfg_feat_keys=None,
            extra_cfg_feat_dims=0,
            graph_embed_dims=1,
            graph_embed_size=1,
        ):
        super().__init__()
        self.model = model
        self.emb = nn.Embedding(128, 128, max_norm=True)
        self.linear_map = nn.Linear(286, 128, bias=True)
        self.op_weights = nn.Parameter(torch.ones(1,1,requires_grad=True) * 100)
        self.config_weights = nn.Parameter(torch.ones(1, 18, requires_grad=True) * 100)
        
        self.graph_embed_dims = graph_embed_dims
        if graph_embed_dims == 1:
            self.history = History(500_000_000, 1)
        else:
            self.history = History(14_000_000, graph_embed_size)
        
        if enc_config:
            self.config_map = nn.Sequential(
                nn.Linear(180, 32, bias=True),
                nn.BatchNorm1d(32),
            )
        if enc_tile_config:
            self.config_map = nn.Sequential(
                nn.Linear(336, 32, bias=True),
                nn.BatchNorm1d(32),
            )
        
        self.extra_cfg_feat_keys = extra_cfg_feat_keys
        if self.extra_cfg_feat_keys:
            self.extra_map = nn.Sequential(
                nn.Linear(extra_cfg_feat_dims, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, 32),
            )
        
        self.input_feat_key = input_feat_key
        self.enc_config = enc_config
        self.enc_tile_config = enc_tile_config
    
    def fourier_enc(self, ten: Tensor, scales=[-1, 0, 1, 2, 3, 4, 5, 6]) -> Tensor:
        """
        ten: (n, feature_dim)
        return: (n, *feature_dim)
        """
        
        def multiscale(x, scales):
            return torch.hstack([x / pow(3., i) for i in scales])
        
        return torch.hstack([
            torch.sin(multiscale(ten, scales)), 
            torch.cos(multiscale(ten, scales))
        ])

    def gather_input_feat(self, batch: Batch) -> Batch:
        batch.split = 'train'
        if self.enc_config:
            config_feats = self.fourier_enc(batch.config_feats, scales=[-1, 0, 1, 2, 3])
            config_feats = self.config_map(config_feats)
        elif self.enc_tile_config:
            config_feats = self.fourier_enc(batch.config_feats, scales=[-1, 0, 1, 2, 3, 4, 5])
            config_feats = self.config_map(config_feats)
        else:
            config_feats = batch.config_feats * self.config_weights
        
        if self.extra_cfg_feat_keys:
            feats = []
            for k in self.extra_cfg_feat_keys:
                scales = [-1, 0, 1, 2, 3, 4, 5, 6, 7] if k == 'extra_read_ops_feat' else [-1, 0, 1, 2, 3, 4, 5]
                fenc = self.fourier_enc(getattr(batch, k), scales=scales)
                feats.append(fenc)
            feats = torch.cat(feats, dim=-1)
            extra_cfg_feats = self.extra_map(feats)
            config_feats = torch.cat([config_feats, extra_cfg_feats], dim=-1)
        
        if self.input_feat_key is None:
            batch.op_emb = self.emb(batch.op_code.long())
            batch.x = torch.cat([
                batch.op_feats, 
                batch.op_emb * self.op_weights,   # TODO: create a per op version of op_weights
                config_feats,
            ], dim=-1)
            batch.x = self.linear_map(batch.x)
        else:
            batch.x = torch.cat([
                getattr(batch, self.input_feat_key),
                config_feats,
            ], dim=-1)
        return batch

    def forward_segment(self, batch: Batch, freeze_body=False):
        custom_gnn = self.model.model  # TPUModel.GraphGymModule.GNN
        module_len = len(list(custom_gnn.children()))
        predict_ctx = torch.no_grad() if freeze_body else nullcontext()

        with predict_ctx:
            for i, module in enumerate(custom_gnn.children()):
                if i < module_len - 1:
                    batch = module(batch)
                if i == module_len - 1:
                    batch_embed = tnn.global_max_pool(batch.x, batch.batch) \
                                    + tnn.global_mean_pool(batch.x, batch.batch)
        
        graph_embed = batch_embed / torch.norm(batch_embed, dim=-1, keepdim=True)
        
        if self.graph_embed_dims == 1:
            for i, module in enumerate(custom_gnn.children()):
                if i == module_len - 1:
                    graph_embed = module.layer_post_mp(graph_embed)
        
        return graph_embed
    
    def join_segments(self, cur_segment: Tensor, batch_other: List[Tensor], batch_num_parts: List[int]) -> Tensor:
        if self.graph_embed_dims == 1:
            return self._join_1d_segments(cur_segment, batch_other, batch_num_parts)
        elif self.graph_embed_dims == 2:
            return self._join_2d_segments(cur_segment, batch_other, batch_num_parts)
        else:
            raise ValueError(f'We only support graph_embed_dims <= 2 for now, but got: {self.graph_embed_dims}')
    
    def _join_1d_segments(self, cur_segment: Tensor, batch_other: List[Tensor], batch_num_parts: List[int]) -> Tensor:
        if batch_other:
            binomial = torch.distributions.binomial.Binomial(probs=0.333)
            """
            Sample some cached embedding of graph segements, 
            use mean of cached + inferenced embedding as full-graph embedding.
            """
            batch_other = torch.cat(batch_other, dim=0)
            mask =  binomial.sample((batch_other.shape[0], 1)).to(torch.device(cfg.device))
            batch_other = batch_other.to(torch.device(cfg.device))
            batch_other = batch_other * mask
            
            batch_other_embed = torch.zeros_like(cur_segment)
            zero_segs = torch.zeros(
                [cur_segment.size(0), 1], 
                dtype=cur_segment.dtype, 
                device=cur_segment.device
            )
            part_cnt = 0
            for i, num_parts in enumerate(batch_num_parts):
                m = num_parts - 1
                batch_other_embed[i, :] += torch.sum(
                    batch_other[part_cnt: part_cnt + m, :],
                    dim=0, 
                    keepdim=False
                )
                zero_segs[i] += (torch.abs(batch_other[part_cnt: part_cnt + m, :]) < 1e-6).sum()
                part_cnt += m
            
            batch_num_parts = torch.Tensor(batch_num_parts).to(torch.device(cfg.device))
            batch_num_parts = batch_num_parts.view(-1, 1)
            # multiplier_num = (batch_num_parts - 1)/ 2 + 1
            multiplier_num = batch_num_parts / (batch_num_parts - zero_segs)
            pred = cur_segment * multiplier_num + batch_other_embed
        else:
            pred = cur_segment
        return pred
    
    def _join_2d_segments(self, cur_segment: Tensor, batch_other: List[Tensor], batch_num_parts: List[int]) -> Tensor:
        if isinstance(batch_other, list):
            batch_other = torch.cat(batch_other, dim=0)
        batch_num_parts = torch.Tensor(batch_num_parts).to(torch.device(cfg.device))
        batch_num_parts = batch_num_parts.view(-1, 1)
        # TODO: maybe adding drop out on "embedding" of batch_other here.
        alpha = 1 / batch_num_parts
        beta = (batch_num_parts - 1) / batch_num_parts
        zero_mask = batch_other.abs().sum(dim=-1) < 1e-6
        alpha[zero_mask] = 1.0
        beta[zero_mask] = 0.0
        pred = alpha * cur_segment + beta * batch_other

        if self.graph_embed_dims == 2:
            custom_gnn = self.model.model  # TPUModel.GraphGymModule.GNN
            module_len = len(list(custom_gnn.children()))
            for i, module in enumerate(custom_gnn.children()):
                if i == module_len - 1:
                    pred = module.layer_post_mp(pred)
        return pred
    
    def update_emb_table(
            self,
            emb: Tensor,
            graph: Data,
            config_id: int,
            segments_idx: int,
        ):
        emb = emb.cpu()
        num_segment = len(graph.partptr) - 1

        if self.graph_embed_dims == 1:
            flat_ind = (
                graph.partition_idx.cpu() + 
                config_id * num_segment + 
                segments_idx
            )
            self.history.push(emb, flat_ind)
        else:
            global_config_id = graph.graph_config_idx
            flat_ind = global_config_id + config_id
            # ema = self.history.pull(flat_ind)
            # ema = ema * ((num_segment - 1) / num_segment) + emb * (1 / num_segment)
            self.history.push(emb, flat_ind)


class CheckpointWrapper:
    """
    A state_dict wrapper act like a nn.Module for a `save_ckpt` method.
    """

    def __init__(self, states):
        self._states = states
    
    def state_dict(self):
        return self._states