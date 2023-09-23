import copy
from typing import *

import torch
import numpy as np
from torch import nn
from torch_sparse import SparseTensor
from torch_geometric.data import Data, Batch

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
        g.config_feats_full[g.config_idx, ...] += g.config_feats
        g.adj = SparseTensor(row=g.edge_index[0], col=g.edge_index[1], sparse_sizes=(g.num_nodes, g.num_nodes))
        processed_batch_list.append(g)
    return Batch.from_data_list(processed_batch_list), sample_idx


def batch_sample_graph_segs(batch_list: List[Data], sampled_idx: torch.Tensor, emb_table: History, num_sample_config=32):
    batch_train_list = []
    batch_other = []
    batch_num_parts = []
    segments_to_train = []
    
    for i in range(len(batch_list)):
        partptr = batch_list[i].partptr.cpu().numpy()
        num_parts = len(partptr) - 1
        batch_num_parts.extend([num_parts] * num_sample_config)
        segment_to_train = np.random.randint(num_parts)
        segments_to_train.append(segment_to_train)
        
        for j in range(num_parts):
            start = int(partptr[j])
            length = int(partptr[j + 1]) - start

            N, E = batch_list[i].num_nodes, batch_list[i].num_edges
            data = copy.copy(batch_list[i])
            del data.num_nodes
            adj, data.adj = data.adj, None

            adj = adj.narrow(0, start, length).narrow(1, start, length)
            edge_idx = adj.storage.value()

            for key, item in data:
                if isinstance(item, torch.Tensor) and item.size(0) == N:
                    data[key] = item.narrow(0, start, length)
                elif isinstance(item, torch.Tensor) and item.size(0) == E:
                    data[key] = item[edge_idx]
                else:
                    data[key] = item

            row, col, _ = adj.coo()
            data.edge_index = torch.stack([row, col], dim=0)
            if j == segment_to_train:
                for k in range(len(data.y)):
                    unfold_g = Data(
                        edge_index=data.edge_index,
                        op_feats=data.op_feats,
                        op_code=data.op_code,
                        config_feats=data.config_feats_full[:, k, :], 
                        num_nodes=length,
                    )
                    batch_train_list.append(unfold_g)
            else:
                for k in range(len(data.y)):
                    batch_other.append(
                        emb_table.pull(
                            batch_list[i].partition_idx.cpu() + num_parts * sampled_idx[i][k] + j
                        )
                    )
    return (
        batch_train_list,
        batch_other,
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
        
        for j in range(num_parts):
            if j == segments_to_train[i]:
                continue
            for k in range(len(data.y)):
                batch_other.append(
                    emb_table.pull(
                        data.partition_idx.cpu() + num_parts * sampled_idx[i][k] + j
                    )
                )
    return batch_other



class TPUModel(torch.nn.Module):
    """
    Wrapper to handle feature embedding/encoding
    """
    def __init__(self, model, input_feat_key=None):
        super().__init__()
        self.model = model
        self.emb = nn.Embedding(128, 128, max_norm=True)
        self.linear_map = nn.Linear(286, 128, bias=True)
        self.op_weights = nn.Parameter(torch.ones(1,1,requires_grad=True) * 100)
        self.config_weights = nn.Parameter(torch.ones(1, 18, requires_grad=True) * 100)
        self.history = History(500000000, 1)
        self.input_feat_key = input_feat_key

    def gather_input_feat(self, batch: Batch) -> Batch:
        batch.split = 'train'
        if self.input_feat_key is None:
            batch.op_emb = self.emb(batch.op_code.long())
            batch.x = torch.cat([
                batch.op_feats, 
                batch.op_emb * self.op_weights,   # TODO: create a per op version of op_weights
                batch.config_feats * self.config_weights
            ], dim=-1)
            batch.x = self.linear_map(batch.x)
        else:
            batch.x = torch.cat([
                getattr(batch, self.input_feat_key),
                batch.config_feats * self.config_weights,
            ], dim=-1)
        return batch