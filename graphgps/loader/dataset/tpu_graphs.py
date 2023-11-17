import copy
import re
import os
import glob
import random
import os.path as osp
from dataclasses import dataclass
from collections import defaultdict
from itertools import product, accumulate
from typing import *

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
from torch import Tensor
from torch_geometric.data import (
  InMemoryDataset,
  Dataset,
  Data, 
  download_url,
  extract_tar, 
  extract_zip
)
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.graphgym.config import cfg
from torch_sparse import SparseTensor


class TPUGraphs(InMemoryDataset):

    def __init__(self, root: str, thres: int = 1000,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 source: str = 'nlp',  # 'nlp' or 'xla'
                 search: str = 'random'  # 'random' or 'default'
                ):
        assert source in ('nlp', 'xla')
        assert search in ('random', 'default')
        self.thres = thres
        self.source = source
        self.search = search
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        op_feats_mean = torch.mean(self.data.op_feats, dim=0, keepdim=True)
        op_feats_std = torch.std(self.data.op_feats, dim=0, keepdim=True)
        op_feats_std[op_feats_std < 1e-6] = 1
        self.data.op_feats = (self.data.op_feats - op_feats_mean) / op_feats_std
        
    @property
    def raw_file_names(self) -> List[str]:
        return [f'npz/layout/{self.source}/{self.search}']

    @property
    def processed_file_names(self) -> List[str]:
        return ['data_segment_{}.pt'.format(self.thres), 'split_dict_segment_{}.pt'.format(self.thres)]

    def process(self):
        """
        * Key "node_config_ids" contains int32 vector with shape (nc, ) and every entry is in {0, 1, ..., n - 1} i.e. indicating the indices of the configurable nodes. 
          For these nodes, they can have an additional feature vector that instructs the compiler (described next).
        * Key "node_config_feat" contains float32 tensor with shape (c, nc, 18). Entry [j, k] gives an 18-dimensional vector describing the configuration features 
          for node d["node_config_ids"][k] for the jth run (please see Subsection "Layout Config Features").
        * Key "config_runtime" contains int32 vector with shape (c, ) where the jth entry contains the runtime of the jth run 
          (i.e., when nodes are configured with d["node_config_feat"][j]).
        """
        data_list = []
        split_names = ['train', 'valid', 'test']
        split_dict = {'train': [], 'valid': [], 'test': []}
        graphs_cnt = 0
        parts_cnt = 0
        for raw_path in self.raw_paths:
            for split_name in split_names:
                filenames = glob.glob(osp.join(os.path.join(raw_path, split_name), '*.npz'))
                print(f' * Process {raw_path} {split_name}')
                for filename in filenames:
                    split_dict[split_name].append(graphs_cnt)
                    np_file = dict(np.load(filename))
                    if "edge_index" not in np_file:
                      print('error in', filename)
                    
                    edge_index = torch.tensor(np_file["edge_index"].T)
                    runtime = torch.tensor(np_file["config_runtime"])
                    op = torch.tensor(np_file["node_feat"])
                    op_code = torch.tensor(np_file["node_opcode"])
                    config_feats = torch.tensor(np_file["node_config_feat"])
                    config_feats = config_feats.view(-1, config_feats.shape[-1])
                    config_idx = torch.tensor(np_file["node_config_ids"])  # node-indies of configurable nodes
                    num_config = torch.tensor(np_file["node_config_feat"].shape[0])
                    num_config_idx = torch.tensor(np_file["node_config_feat"].shape[1])  # number of configurable nodes
                    
                    num_nodes = torch.tensor(np_file["node_feat"].shape[0])
                    num_parts = num_nodes // self.thres + 1
                    interval = num_nodes // num_parts
                    partptr = torch.arange(0, num_nodes, interval+1)  # global id of the graph segments
                    if partptr[-1] != num_nodes:
                        partptr = torch.cat([partptr, torch.tensor([num_nodes])])
                    
                    data = Data(edge_index=edge_index, op_feats=op, op_code=op_code, 
                                config_feats=config_feats, config_idx=config_idx,
                                num_config=num_config, num_config_idx=num_config_idx, y=runtime, 
                                num_nodes=num_nodes, partptr=partptr, partition_idx = parts_cnt)
                    data_list.append(data)
                    graphs_cnt += 1
                    parts_cnt += num_parts * num_config
            
            if not data_list:
              raise RuntimeError(f"Can't find any dataset samples in: {self.raw_paths}")
            torch.save(self.collate(data_list), self.processed_paths[0])
            torch.save(split_dict, self.processed_paths[1])
    
    def get_idx_split(self):
        return torch.load(self.processed_paths[1])


@dataclass
class DatasetStatistics:
    op_feat_mean: torch.Tensor
    op_feat_std: torch.Tensor
    num_nodes: int
    num_graphs: int
    num_graph_configs: int
    num_segments: int
    num_unique_segments: int
    max_node_per_graph: int



class BasicSampler:
    """
    random sample configs and (try to )remove duplication
    """

    def resample(self, graph: Data, num_sample_configs: int) -> Tuple[Tensor]:
        num_config = graph.num_config.item()
        all_zero = (graph.y < 1e-9).all()
        pad_mask = torch.zeros([num_sample_configs], dtype=torch.bool)
        if all_zero:  # runtime can't be sort
            return torch.randint(0, num_config, (num_sample_configs,)), pad_mask
        
        randperm = torch.randperm(num_config)
        sample_idx = randperm[:num_sample_configs]
        sample_cfg = graph.config_feats.view(graph.num_config, graph.num_config_idx, -1)
        sample_cfg = sample_cfg[sample_idx, ...]

        resample_ptr = num_sample_configs
        for i, cfeat in enumerate(sample_cfg):
            delta = torch.abs(sample_cfg[i + 1:] - cfeat).sum(axis=-1).sum(axis=-1)
            mask = delta < 1e-6
            if mask.any():
                # NOTE: overwrite duplicated samples with same id, so at least the label will also be the same 
                # if we cant find the alternative samples, and we can exclue the duplicans in loss func by looking at label.
                sample_idx[i + 1:][mask] = sample_idx[i]
                pad_mask[i + 1:i + 1 + mask.size(0)][mask] = True
            if resample_ptr + mask.sum() >= num_config:
                break
            if mask.any():
                sample_idx[i + 1:][mask] = randperm[resample_ptr: resample_ptr + mask.sum()]
                pad_mask[i + 1:][mask] = False
                resample_ptr += mask.sum()
        
        if len(sample_idx) < num_sample_configs:
            miss = num_sample_configs - len(sample_idx)
            pad = torch.zeros([miss], dtype=sample_idx.dtype, device=sample_idx.device) + sample_idx[0]
            sample_idx = torch.cat([sample_idx, pad], dim=0)
            pad_mask[-miss:] = True
        
        return sample_idx, pad_mask


class IntervalSampler(BasicSampler):
    """
    Each graph can have upto tens thoughs of config, and each config will have one runtime as the groundtruth ref of the model.
    All the runtimes from a single graph can cover a large ranges, ex: a graph can have 4e7 ~ 1e9 ms runtime according to different config,
    And this sampler pick only some subset that have similiary runtimes that help model to learn more fine-grain different
    between different configs.
    """
    def __init__(self, interval_size=512, interval_lifetime=1) -> None:
        self.interval_size = interval_size  # NOTE: this should be >= cfg.dataset.num_sample_config
        self.interval_lifetime = interval_lifetime
        self.lifetimes = defaultdict(lambda: 0)
        self.intervals = {}
    
    def resample(self, graph: Data, num_sample_configs: int) -> Tuple[Tensor]:
        all_zero = (graph.y < 1e-9).all()
        pad_mask = torch.zeros([num_sample_configs], dtype=torch.bool)
        if all_zero or random.random() < 0.8:  # runtime can't be sort
            return super().resample(graph, num_sample_configs)
        
        ind = f"{graph.graph_name}_{graph.source_dataset}"
        if self.lifetimes[ind] <= 0:
            n = graph.y.size(0)
            _, indices = graph.y.sort()
            low = random.randint(0, max(0, n - 1 - self.interval_size))
            hi = min(n - 1, low + self.interval_size)
            self.intervals[ind] = indices[low: hi]
            self.lifetimes[ind] = self.interval_lifetime
        else:
            self.lifetimes[ind] -= 1
        
        # resample_idx = torch.randint(0, len(self.intervals[ind]), [num_sample_configs])
        resample_idx = torch.randperm(len(self.intervals[ind]))
        sample_idx = self.intervals[ind][resample_idx[:num_sample_configs]]
        
        sample_cfg = graph.config_feats.view(graph.num_config, graph.num_config_idx, -1)
        sample_cfg = sample_cfg[sample_idx, ...]
        resample_ptr = num_sample_configs
        for i, cfeat in enumerate(sample_cfg):
            delta = torch.abs(sample_cfg[i + 1:] - cfeat).sum(axis=-1).sum(axis=-1)
            mask = delta < 1e-6
            if mask.any():
                sample_idx[i + 1:][mask] = sample_idx[i]
                pad_mask[i + 1:i + 1 + mask.size(0)][mask] = True
            if resample_ptr + mask.sum() >= len(self.intervals[ind]):
                break
            if mask.any():
                sample_idx[i + 1:][mask] = self.intervals[ind][resample_ptr: resample_ptr + mask.sum()]
                pad_mask[i + 1:][mask] = False
                resample_ptr += mask.sum()
        
        if len(sample_idx) < num_sample_configs:
            mis = num_sample_configs - len(sample_idx)
            pad = torch.zeros([mis], dtype=sample_idx.dtype, device=sample_idx.device) + sample_idx[0]
            sample_idx = torch.cat([sample_idx, pad], dim=0)
            pad_mask[-mis:] = True
        return sample_idx, pad_mask


class KeepKHop:

    def __init__(self, hops=2, bidirect=False) -> None:
        """
        bidirect: 
            find the k-hop neighbor on a birectional version of the graph, but not actually modify the graph
        """
        self.hops = hops
        self.bidirect = bidirect
    
    def __call__(self, graph: Data) -> Data:
        assert hasattr(graph, 'config_idx')
        src_edge_idx = graph.edge_index
        if self.bidirect:
            src_edge_idx = torch.cat([
                    src_edge_idx, 
                    torch.stack([src_edge_idx[1, :], src_edge_idx[0, :]], dim=0),
                ], 
                dim=1
            )
        
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(
            graph.config_idx, 
            self.hops, 
            src_edge_idx, 
            relabel_nodes=True
        )
        if self.bidirect:
            num_edge = edge_index.size(1)
            graph.edge_index = edge_index[:, :num_edge // 2]
        else:
            graph.edge_index = edge_index
        # NOTE: keep config_feats untouch, since the relative order of confiurable nodes is the same in new graph.
        graph.config_idx = mapping
        graph.num_config_idx = torch.tensor(len(mapping))
        graph.num_nodes = torch.tensor(len(subset))
        graph.partptr = torch.cat(
            [
                graph.partptr[graph.partptr < graph.num_nodes], 
                torch.tensor([graph.num_nodes])
            ], 
            dim=0
        )
        for k in ['pestat_RWSE','eigvecs_sn', 'eigvals_sn', 'op_feats', 'op_code', 'op_feat_enc_i']:
            if hasattr(graph, k):
                val = getattr(graph, k)
                setattr(graph, k, val[subset])
        
        return graph


class TPUGraphsNpz(Dataset):

    KEYS = [
        'num_config_idx', 'pestat_RWSE', 'edge_index', 'config_feats', 
        'num_config', 'eigvecs_sn', 'config_idx', 'op_feats', 'y', 
        'num_nodes', 'partptr', 'partition_idx', 'op_code', 'eigvals_sn', 'graph_name', 'source_dataset',
    ]
    EXTRA =  ["extra_feat", "extra_read_ops_feat"]
    
    def __init__(
          self, 
          root: str, 
          thres: int = 1000,
          transform: Optional[Callable] = None,
          pre_transform: Optional[Callable] = None,
          pre_filter: Optional[Callable] = None,
          source: str = 'nlp',  # 'nlp' or 'xla'
          search: str = 'random',  # 'random' or 'default'
          task: str = 'layout',
          cache_in_memory: bool = False,
        ):
        assert source in ('nlp', 'xla')
        assert search in ('random', 'default')
        print(f'[TPUGraphsNpz] source: {source}, search: {search}, cache_in_memory: {cache_in_memory}')
        self.thres = thres
        self.source = source
        self.search = search
        self.task = task
        self.epoch_multiply = 1
        self.cache_in_memory = cache_in_memory
        self._cache = {}
        self._cache_limit = 8000
        self._norm_op_feat = False
        super().__init__(root, transform, pre_transform, pre_filter)
        if self.cache_in_memory:
            # for i in tqdm(range(len(self)), desc='Pre-Caching'):
            #     self[i]
            self.meta
        self.data = Data(
            edge_index=None,
            op_feats=None,
            op_code=None,
            config_feats=None,
            config_idx=None,
            num_config=None,
            num_config_idx=None,
            y=None,
            partptr=None,
            partition_idx=None,
            num_nodes=1,
        )
        self.slices = None
        self.label_sampler = None
    
    @property
    def meta(self):
        if not hasattr(self, "_meta"):
            op_feats = []
            print('Computing meta...')
            save_file = os.path.join(self.processed_dir, f"{self.source}_{self.search}_meta.pt")
            
            if os.path.exists(save_file):
                self._meta = torch.load(save_file)
                return self._meta
            
            total_nodes = 0
            total_unq_segs = 0
            total_segs = 0
            total_graphs = 0
            total_cfgs = 0
            max_nodes = 0
            
            idx_split = self.get_idx_split()
            for ind in tqdm(idx_split['train']):
                path = self.processed_paths[ind]
                data = torch.load(path)
                if isinstance(data, Data):
                    op_feats.append(data.op_feats)
                    num_node = data.op_feats.size(0)
                    num_cfgs = len(data.y)
                    # num_cfgs = data.num_config
                    total_cfgs += num_cfgs
                    total_nodes += num_node
                    total_unq_segs += num_node // self.thres + 1
                    total_segs += (num_node // self.thres + 1) * num_cfgs
                    max_nodes = max(max_nodes, num_node)
                    total_graphs += 1
            
            op_feats = torch.concat(op_feats, dim=0)
            op_feats_mean = torch.mean(op_feats, dim=0, keepdim=True)
            op_feats_std = torch.std(op_feats, dim=0, keepdim=True)
            op_feats_std[op_feats_std < 1e-6] = 1
            
            self._meta = DatasetStatistics(
                op_feat_mean=op_feats_mean,
                op_feat_std=op_feats_std,
                num_graphs=total_graphs,
                num_graph_configs=total_cfgs,
                num_nodes=total_nodes,
                num_segments=total_segs,
                num_unique_segments=total_unq_segs,
                max_node_per_graph=max_nodes,
            )
            torch.save(self._meta, save_file)
            print(self._meta)
        # self.data.op_feats = (self.data.op_feats - op_feats_mean) / op_feats_std
        return self._meta
        
    @property
    def raw_file_names(self):
        if self.task == "layout":
            pattern = osp.join(self.raw_dir, f"npz/layout/{self.source}/{self.search}/**", '*.npz')
        else:
            pattern = osp.join(self.raw_dir, f"npz/tile/xla/**", '*.npz')
        raw_dir = self.raw_dir
        if not raw_dir.endswith(osp.sep):
            raw_dir += osp.sep
        relative_paths = [p.replace(raw_dir, "") for p in glob.glob(pattern, recursive=True)]
        return relative_paths

    @property
    def processed_file_names(self):
        if self.task == "layout":
            pattern = osp.join(self.processed_dir, f'{self.source}_{self.search}_data_*.pt')
            exist_files = len(glob.glob(pattern))
            target_size = len(self.raw_file_names)
            files = [
                f'{self.source}_{self.search}_data_{i}.pt' 
                for i in range(max(exist_files, target_size))
            ]
            files.append(f'{self.source}_{self.search}_split_dict.pt')
        else:
            files = [f'xla_{self.task}_data_{i}.pt' for i in range(len(self.raw_file_names))]
            files.append(f'xla_{self.task}_split_dict.pt')
        return files

    def process(self):
        split_names = ['train', 'valid', 'test']
        split_dict = {'train': [], 'valid': [], 'test': []}
        parts_cnt = 0
        config_counts = 0
        
        for idx, raw_path in enumerate(tqdm(self.raw_paths)):
            if self.task == 'layout':
                out_path = osp.join(self.processed_dir, f'{self.source}_{self.search}_data_{idx}.pt')
            else:
                out_path = osp.join(self.processed_dir, f'xla_{self.task}_data_{idx}.pt')
            split_name = osp.basename(osp.dirname(raw_path))
            split_dict[split_name].append(idx)

            if osp.exists(out_path):
                old_data = torch.load(out_path)
                if not set(old_data.keys).symmetric_difference(set(self.KEYS)):
                    print("SKIP ", out_path)
                    continue
            
            np_file = dict(np.load(raw_path))
            if "edge_index" not in np_file:
                print('error in', raw_path)
            
            num_nodes = torch.tensor(np_file["node_feat"].shape[0])
            num_parts = num_nodes // self.thres + 1
            interval = num_nodes // num_parts
            # NOTE: node_id within [partptr[i], partptr[i + 1]) is belong to graph-segment-i
            partptr = torch.arange(0, num_nodes, interval+1)  # TODO: Find a better way to partition graph according to topologic 
            if partptr[-1] != num_nodes:
                partptr = torch.cat([partptr, torch.tensor([num_nodes])])
            graph_name = osp.basename(raw_path).replace('.npz', '')
            
            edge_index = torch.tensor(np_file["edge_index"].T)
            runtime = torch.tensor(np_file["config_runtime"])
            op = torch.tensor(np_file["node_feat"])
            op_code = torch.tensor(np_file["node_opcode"])

            if self.task == 'layout':
                config_feats = torch.tensor(np_file["node_config_feat"])  # (c, nc, 18)
                config_feats = config_feats.view(-1, config_feats.shape[-1])
                config_idx = torch.tensor(np_file["node_config_ids"])  # node-indies of configurable nodes
                num_config = torch.tensor(np_file["node_config_feat"].shape[0])
                num_config_idx = torch.tensor(np_file["node_config_feat"].shape[1])  # number of configurable nodes
            else:
                config_feats = torch.tensor(np_file["config_feat"])  # (c, 24)
                config_feats = config_feats.unsqueeze(dim=1).repeat([1, num_nodes, 1])
                config_feats = config_feats.view(-1, config_feats.shape[-1])
                config_idx = torch.arange(0, num_nodes)
                num_config = torch.tensor(np_file["config_feat"].shape[0])
                num_config_idx = torch.tensor(num_nodes)  # number of configurable nodes

            if -2**7 <= config_feats.min() and config_feats.max() <= 2**7:
                config_feats = config_feats.to(torch.int8)
            elif -2**15 <= config_feats.min() and config_feats.max() <= 2**15:
                config_feats = config_feats.to(torch.int16)
            
            data = Data(edge_index=edge_index, op_feats=op, op_code=op_code, 
                        config_feats=config_feats, config_idx=config_idx,
                        num_config=num_config, num_config_idx=num_config_idx, y=runtime, 
                        num_nodes=num_nodes, partptr=partptr, partition_idx=parts_cnt, 
                        graph_name=graph_name, graph_config_idx=config_counts, graph_idx=idx)
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            parts_cnt += num_parts * num_config
            config_counts += num_config
            torch.save(data, out_path)
        torch.save(split_dict, self.processed_paths[-1])

    def len(self):
        n = len(self.processed_file_names) - 1
        return n * self.epoch_multiply

    def get(self, idx):
        if idx in self._cache:
            # print('take from cache ', idx)
            data = copy.deepcopy(self._cache[idx])
        else:
            if self.task == 'layout':
                pt_file = osp.join(self.processed_dir, f'{self.source}_{self.search}_data_{idx}.pt')
            else:
                pt_file = osp.join(self.processed_dir, f'xla_{self.task}_data_{idx}.pt')
            data = torch.load(pt_file)
            if isinstance(data.partition_idx, int):  # HACK: habdle the case that PyGemo not able to convert int to tensor
                data.partition_idx = torch.tensor(data.partition_idx)
            
            if self._norm_op_feat:
                op_feats_mean = self.meta.op_feat_mean
                op_feats_std = self.meta.op_feat_std
                data.op_feats = (data.op_feats - op_feats_mean) / op_feats_std
            
            for key in self.EXTRA:
                if key in data.keys and (key not in cfg.dataset.extra_cfg_feat_keys):
                    delattr(data, key)
            
            if self.cache_in_memory and len(self._cache) < self._cache_limit:
                self._cache[idx] = copy.deepcopy(data) #.share_memory_()
        
        data.config_feats = data.config_feats.float()
        data.source_dataset = f"{self.source}-{self.search}-{idx}" if self.task == 'layout' else f"xla-tile-{idx}"
        data.submit_id = f"{self.task}:{self.source}:{self.search}:{data.graph_name}"

        if self.transform:
            data = self.transform(data)
        
        if cfg.train.regression.val_min >= 0:
            data.y = data.y.float()
            data.y -= cfg.train.regression.val_min
            if cfg.train.regression.val_max > cfg.train.regression.val_min:
                scope = cfg.train.regression.val_max - cfg.train.regression.val_min
                data.y = (data.y / scope) * 100
        
        if hasattr(data, 'graph_idx'):
            delattr(data, 'graph_idx')
            delattr(data, 'graph_config_idx')
        return data
    
    def get_idx_split(self):
        if not hasattr(self, "_split_idxs"):
            self._split_idxs = torch.load(self.processed_paths[-1])
        return self._split_idxs



class MixTPUGraphsNpz(Dataset):
    
    def __init__(
          self, 
          root: str, 
          thres: int = 1000,
          transform: Optional[Callable] = None,
          pre_transform: Optional[Callable] = None,
          pre_filter: Optional[Callable] = None,
          source: str = 'nlp+xla',  # 'nlp' or 'xla'
          search: str = 'random+default',  # 'random' or 'default'
          cache_in_memory: bool = False,
          valid_for_train: List[str] = [],
        ):
        source: List[str] = sorted(source.split('+'))
        search: List[str] = sorted(search.split('+'))
        self.dataset_names: List[str] = []
        self.datasets: Dict[str, TPUGraphsNpz] = {}
        
        for a, b in product(source, search):
            name = f"{a}_{b}"
            self.dataset_names.append(name)
            self.datasets[name] = TPUGraphsNpz(
                root.replace('MixTPUGraphsNpz', 'TPUGraphsNpz'),
                thres=thres,
                transform=transform,
                pre_transform=pre_transform,
                pre_filter=pre_filter,
                source=a,
                search=b,
                cache_in_memory=cache_in_memory,
            )
        self.dataset_weights = [1 if 'val' in name else 2 for name in self.dataset_names]
        _wmap = {k: v for k, v in zip(self.dataset_names, self.dataset_weights)}
        logger.info(f"dataset_weights: {_wmap}")
        
        self.valid_for_train = valid_for_train
        self.custom_split_names = ['train'] 
        self.custom_split_names += [
            f'valid_{v}' 
            for v in self.dataset_names 
            if v not in self.valid_for_train
        ]
        self.custom_split_names += ['test']  # for split_generator.py
        logger.info(f"dataset_weights: {self.custom_split_names}")
        super().__init__(root, transform, pre_transform, pre_filter)
        
        # HACK: dummy for passing graphgps dataset check
        self.data = Data(
            edge_index=None,
            op_feats=None,
            op_code=None,
            config_feats=None,
            config_idx=None,
            num_config=None,
            num_config_idx=None,
            y=None,
            partptr=None,
            partition_idx=None,
            num_nodes=1,
        )
        self.slices = None
    
    @property
    def segment_offsets(self):
        offsets = [0]
        for k in self.dataset_names[:-1]:
            prev = offsets[-1]
            offsets.append(prev + self.datasets[k].meta.num_unique_segments)
        return offsets
    
    @property
    def label_sampler(self):
        return [self.datasets[k].label_sampler for k in self.dataset_names]
    
    @label_sampler.setter
    def label_sampler(self, sampler):
        for dataset in self.datasets.values():
            dataset.label_sampler = copy.deepcopy(sampler)

    def len(self):
        sizes = [len(dset) for dset in self.datasets.values()]
        return sum(sizes)

    def _get(self, idx):
        end_indies = [0] + list(accumulate(len(self.datasets[k]) for k in self.dataset_names))
        for i, (a, b) in enumerate(zip(end_indies, end_indies[1:])):
            if a <= idx < b:
                src = self.datasets[self.dataset_names[i]]
                graph: Data = src.get(idx - a)
                if hasattr(graph, 'partition_idx'):
                    segment_offset = self.segment_offsets[i]
                    graph.partition_idx += segment_offset
                return graph
        raise IndexError(f"{idx} isn't a valid index in a dataset of size {len(self)}")

    def get(self, idx):
        if hasattr(self, 'split_name') and 'train' in self.split_name:
            if not self.valid_for_train:
                # i, src_name = random.choice(list(enumerate(self.dataset_names)))
                i, src_name = random.choices(
                    list(enumerate(self.dataset_names)), 
                    weights=self.dataset_weights, 
                    k=1
                )[0]
                src = self.datasets[src_name]
                sub_id = random.randint(0, len(src) - 1)
                graph: Data = src.get(sub_id)

                if hasattr(graph, 'partition_idx'):
                    segment_offset = self.segment_offsets[i]
                    graph.partition_idx += segment_offset
            else:
                graph: Data = self._get(idx)
            return graph
        else:
            return self._get(idx)
    
    def get_idx_split(self):
        if not hasattr(self, "_split_idxs"):
            self._split_idxs = defaultdict(list)
            start_indies = [0] + list(accumulate(len(self.datasets[k]) for k in self.dataset_names[:-1]))
            start_indies = { k: v for k, v in zip(self.dataset_names, start_indies) }
            val_for_train_idx = []
            
            for name, dataset in self.datasets.items():
                offset = start_indies[name]
                for split, idx in dataset.get_idx_split().items():
                    off_idx = [j + offset for j in idx]
                    if split == 'valid':
                        if name in self.valid_for_train:
                            val_for_train_idx += off_idx
                        else:
                            self._split_idxs[f"valid_{name}"] = off_idx
                    else:
                        self._split_idxs[split] += off_idx
            
            if val_for_train_idx:
                val_size = len(val_for_train_idx)
                full_train = self._split_idxs['train']
                random.shuffle(full_train)
                logger.warning(f"Remixing train dataset [{len(full_train)}] with val set[{val_size}] for training")
                self._split_idxs['train'] = val_for_train_idx + full_train[:val_size * 5]
        return self._split_idxs


if __name__ == '__main__':
    dataset = TPUGraphs(root='datasets/TPUGraphs')
    import pdb; pdb.set_trace()
