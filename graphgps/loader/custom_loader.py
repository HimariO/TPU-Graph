import copy
from typing import Callable, List
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch_geometric.graphgym.register as register
import torch_geometric.transforms as T
from torch_geometric.datasets import (
    PPI,
    Amazon,
    Coauthor,
    KarateClub,
    MNISTSuperpixels,
    Planetoid,
    QM7b,
    TUDataset,
)
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.transform import (
    create_link_label,
    neg_sampling_transform,
)
from torch_geometric.loader import (
    ClusterLoader,
    GraphSAINTEdgeSampler,
    GraphSAINTNodeSampler,
    GraphSAINTRandomWalkSampler,
    NeighborSampler,
    RandomNodeLoader,
)
from torch_geometric.utils import (
    index_to_mask,
    negative_sampling,
    to_undirected,
)
from torch_geometric.graphgym.loader import (
    load_pyg,
    set_dataset_attr,
    planetoid_dataset,
    load_ogb,
    load_dataset,
    set_dataset_info,
    create_dataset,
)
from torch_geometric.data import Batch, Data
from torch_sparse import SparseTensor
from graphgps.loader.dataset.tpu_graphs import IntervalSampler

index2mask = index_to_mask  # TODO Backward compatibility


def batch_sample_graph_segs(batch: Batch, num_sample_config=32):
    # HACK: doing the reduant `to_data_list` here so every tensor in Data will be at least 1D
    batch_list = batch.to_data_list()
    batch_train_list = []
    # batch_other = []
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
            if j == segment_to_train:
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
        # batch_other,
        batch_num_parts,
        segments_to_train,
    )


def preprocess_batch(batch, num_sample_configs=32, train_graph_segment=False, sampler=None):
    
    # batch_list = batch.to_data_list()
    batch_list = batch
    processed_batch_list = []
    sample_idx = []
    max_config_num = max(g.num_config.item() for g in batch_list)
    
    for g in batch_list:
        if train_graph_segment:
            if sampler is None:
                sample_idx.append(
                    torch.randint(0, g.num_config.item(), (num_sample_configs,))
                )
            else:
                sample_idx.append(sampler.resample(g, num_sample_configs))
        else:
            sample_idx.append(
                # torch.arange(0, min(g.num_config.item(), num_sample_configs))
                torch.arange(0, min(max_config_num, num_sample_configs)) % g.num_config.item()
            )
        g.y = g.y[sample_idx[-1]]
        g.config_feats = g.config_feats.view(g.num_config, g.num_config_idx, -1)[sample_idx[-1], ...]
        g.config_feats = g.config_feats.transpose(0,1)
        # NOTE: add padding to non-configable nodes
        g.config_feats_full = torch.zeros(
            [
                g.num_nodes,
                len(sample_idx[-1]),
                g.config_feats.shape[-1]
            ], 
            device=g.config_feats.device
        )
        g.config_feats_full -= 1
        g.config_feats_full[g.config_idx, ...] = g.config_feats

        for feat_key in cfg.dataset.extra_cfg_feat_keys:
            extra_feat = getattr(g, feat_key).float()
            extra_feat = extra_feat[sample_idx[-1], ...].transpose(0, 1)
            full_feat = torch.zeros(
                [
                    g.num_nodes,
                    len(sample_idx[-1]),
                    extra_feat.shape[-1]
                ], 
                device=extra_feat.device
            )
            full_feat -= 1
            full_feat[g.config_idx, ...] = extra_feat
            setattr(g, feat_key, full_feat)
        
        g.adj = SparseTensor(row=g.edge_index[0], col=g.edge_index[1], sparse_sizes=(g.num_nodes, g.num_nodes))
        processed_batch_list.append(g)
    # breakpoint()
    processed_batch_list = Batch.from_data_list(processed_batch_list)
    if train_graph_segment:
        return (
            batch_sample_graph_segs(processed_batch_list, num_sample_config=num_sample_configs), 
            sample_idx
        )
    else:
        return processed_batch_list, sample_idx


def get_loader(dataset, sampler, batch_size, shuffle=True, train=False):
    if sampler == "full_batch" or len(dataset) > 1:
        config_sampler = {
            'IntervalSampler': IntervalSampler,
        }
        if train and cfg.dataset.config_sampler in config_sampler:
            sampler = config_sampler[cfg.dataset.config_sampler]()
        else:
            sampler = None
        
        collate_fn = (
            partial(
                preprocess_batch,
                train_graph_segment=True,
                num_sample_configs=cfg.dataset.num_sample_config,
                sampler=sampler,
            ) 
            if train else 
            partial(
                preprocess_batch,
                num_sample_configs=cfg.dataset.eval_num_sample_config
            ) 
        )
        
        
        loader_train = DataLoader(dataset, batch_size=batch_size,
                                  shuffle=shuffle, 
                                  num_workers=cfg.num_workers,
                                  pin_memory=False, 
                                  persistent_workers=cfg.num_workers > 0,
                                  collate_fn=collate_fn)
    elif sampler == "neighbor":
        loader_train = NeighborSampler(
            dataset[0], sizes=cfg.train.neighbor_sizes[:cfg.gnn.layers_mp],
            batch_size=batch_size, shuffle=shuffle,
            num_workers=cfg.num_workers, pin_memory=True)
    elif sampler == "random_node":
        loader_train = RandomNodeLoader(dataset[0],
                                        num_parts=cfg.train.train_parts,
                                        shuffle=shuffle,
                                        num_workers=cfg.num_workers,
                                        pin_memory=True)
    elif sampler == "saint_rw":
        loader_train = \
            GraphSAINTRandomWalkSampler(dataset[0],
                                        batch_size=batch_size,
                                        walk_length=cfg.train.walk_length,
                                        num_steps=cfg.train.iter_per_epoch,
                                        sample_coverage=0,
                                        shuffle=shuffle,
                                        num_workers=cfg.num_workers,
                                        pin_memory=True)
    elif sampler == "saint_node":
        loader_train = \
            GraphSAINTNodeSampler(dataset[0], batch_size=batch_size,
                                  num_steps=cfg.train.iter_per_epoch,
                                  sample_coverage=0, shuffle=shuffle,
                                  num_workers=cfg.num_workers,
                                  pin_memory=True)
    elif sampler == "saint_edge":
        loader_train = \
            GraphSAINTEdgeSampler(dataset[0], batch_size=batch_size,
                                  num_steps=cfg.train.iter_per_epoch,
                                  sample_coverage=0, shuffle=shuffle,
                                  num_workers=cfg.num_workers,
                                  pin_memory=True)
    elif sampler == "cluster":
        loader_train = \
            ClusterLoader(dataset[0],
                          num_parts=cfg.train.train_parts,
                          save_dir="{}/{}".format(cfg.dataset.dir,
                                                  cfg.dataset.name.replace(
                                                      "-", "_")),
                          batch_size=batch_size, shuffle=shuffle,
                          num_workers=cfg.num_workers,
                          pin_memory=True)

    else:
        raise NotImplementedError("%s sampler is not implemented!" % sampler)
    return loader_train


def create_loader():
    """
    Create data loader object

    Returns: List of PyTorch data loaders

    """
    dataset = create_dataset()
    # train loader
    if cfg.dataset.task == 'graph':
        id = dataset.data['train_graph_index']
        loaders = [
            get_loader(dataset[id], cfg.train.sampler, cfg.train.batch_size,
                       shuffle=True, train=True)
        ]
        loaders[-1].dataset.split_name = 'train'
        delattr(dataset.data, 'train_graph_index')
    else:
        loaders = [
            get_loader(dataset, cfg.train.sampler, cfg.train.batch_size,
                       shuffle=True, train=True)
        ]

    if hasattr(dataset, 'custom_split_names'):
        for i in range(1, len(dataset.custom_split_names)):
            if cfg.dataset.task == 'graph':
                split_names = [f'{n}_graph_index' for n in dataset.custom_split_names]
                indies = dataset.data[split_names[i]]
                loaders.append(
                    get_loader(
                        dataset[indies],
                        cfg.val.sampler,
                        cfg.train.batch_size,
                        shuffle=False
                    )
                )
                split_names = dataset.custom_split_names
                loaders[-1].dataset.split_name = split_names[i]
                delattr(dataset.data, split_names[i])
            else:
                raise NotImplementedError()
    else:
        # val and test loaders
        for i in range(cfg.share.num_splits - 1):
            if cfg.dataset.task == 'graph':
                split_names = ['val_graph_index', 'test_graph_index']
                id = dataset.data[split_names[i]]
                loaders.append(
                    get_loader(dataset[id], cfg.val.sampler, cfg.train.batch_size,
                            shuffle=False))
                split_names = ['valid', 'test']
                loaders[-1].dataset.split_name = split_names[i]
                delattr(dataset.data, split_names[i])
            else:
                loaders.append(
                    get_loader(dataset, cfg.val.sampler, cfg.train.batch_size,
                            shuffle=False))

    return loaders
