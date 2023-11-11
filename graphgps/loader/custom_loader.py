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
from graphgps.loader.dataset.tpu_graphs import IntervalSampler, BasicSampler
from graphgps.train.gst_utils import (
    batch_sample_graph_segs, 
    batch_sample_full,
    select_graph_config,
    form_config_pair
)

index2mask = index_to_mask  # TODO Backward compatibility


def preprocess_batch(batch, num_sample_configs=32, train_graph_segment=False, sampler=None, full_graph=False):
    
    # batch_list = batch.to_data_list()
    batch_list = batch
    processed_batch_list = []
    sample_idx = []
    max_config_num = max(g.num_config.item() for g in batch_list)
    
    for g in batch_list:
        padding_mask = torch.zeros([num_sample_configs], dtype=torch.bool)
        if train_graph_segment:
            if sampler is None:
                sample_idx.append(
                    torch.randint(0, g.num_config.item(), (num_sample_configs,))
                )
            else:
                idx, padding_mask = sampler.resample(g, num_sample_configs)
                sample_idx.append(idx)
        else:
            sample_idx.append(
                # torch.arange(0, min(g.num_config.item(), num_sample_configs))
                torch.arange(0, min(max_config_num, num_sample_configs)) % g.num_config.item()
            )
        
        g = select_graph_config(g, sample_idx, padding_mask, train_graph_segment)
        if cfg.train.pair_rank:
            g = form_config_pair(g, train=train_graph_segment)
        processed_batch_list.append(g)

    processed_batch_list = Batch.from_data_list(processed_batch_list)
    if train_graph_segment:
        if full_graph:
            return (
                batch_sample_full(processed_batch_list, num_sample_config=num_sample_configs), 
                sample_idx
            )
        else:
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
            sampler = BasicSampler()
        
        collate_fn = (
            partial(
                preprocess_batch,
                train_graph_segment=True,
                num_sample_configs=cfg.dataset.num_sample_config,
                full_graph=cfg.train.gst.sample_full_graph,
                sampler=sampler,
            ) 
            if train else 
            partial(
                preprocess_batch,
                num_sample_configs=cfg.dataset.eval_num_sample_config,
                full_graph=cfg.train.gst.sample_full_graph,
            ) 
        )
        
        if cfg.num_workers > 0:
            prefetch_factor = cfg.prefetch_factor if train else (1 if cfg.train.pair_rank else 2)
        else:
            prefetch_factor = None
        loader_train = DataLoader(dataset, batch_size=batch_size,
                                  shuffle=shuffle, 
                                  num_workers=cfg.num_workers,
                                  pin_memory=False, 
                                  persistent_workers=cfg.num_workers > 0,
                                  prefetch_factor=prefetch_factor,
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
