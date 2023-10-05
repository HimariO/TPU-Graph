import copy
import re
import os
import glob
import random
import os.path as osp
from dataclasses import dataclass
from collections import defaultdict
from itertools import product, accumulate
from typing import Optional, Callable, List, Dict

import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import (
  InMemoryDataset,
  Dataset,
  Data, 
  download_url,
  extract_tar, 
  extract_zip
)
from torch_geometric.utils import remove_isolated_nodes
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
    num_segments: int
    num_unique_segments: int
    max_node_per_graph: int


class IntervalSampler:
    """
    Each graph can have upto tens thoughs of config, and each config will have one runtime as the groundtruth ref of the model.
    All the runtimes from a single graph can cover a large ranges, ex: a graph can have 4e7 ~ 1e9 ms runtime according to different config,
    And this sampler pick only some subset that have similiary runtimes that help model to learn more fine-grain different
    between different configs.
    """
    def __init__(self, interval_size=512, interval_lifetime=40) -> None:
        self.interval_size = interval_size  # NOTE: this should be >= cfg.dataset.num_sample_config
        self.interval_lifetime = interval_lifetime
        self.lifetimes = defaultdict(lambda: 0)
        self.intervals = {}
    
    def _resample(self, ind: int, graph: Data):
        all_zero = (graph.y < 1e-6).all()
        if all_zero:  # runtime can't be sort
            return graph
        
        if self.lifetimes[ind] <= 0:
            n = graph.y.size(0)
            _, indices = graph.y.sort()
            low = random.randint(0, max(0, n - 1 - self.interval_size))
            hi = min(n - 1, low + self.interval_size)
            self.intervals[ind] = indices[low: hi]
            self.lifetimes[ind] = self.interval_lifetime
        else:
            self.lifetimes[ind] -= 1
        
        new_graph = copy.deepcopy(graph)
        new_graph.y = new_graph.y[self.intervals[ind]]
        
        feat_dim = new_graph.config_feats.size(-1)
        new_graph.config_feats = new_graph.config_feats.view(
            new_graph.num_config, -1, feat_dim)
        new_graph.config_feats = new_graph.config_feats[self.intervals[ind]]
        new_graph.config_feats = new_graph.config_feats.view(-1, feat_dim)
        
        new_graph.num_config = new_graph.y.size(0)
        return new_graph
    
    def resample(self, graph: Data, num_sample_configs: int):
        all_zero = (graph.y < 1e-6).all()
        # if all_zero or random.random() < 0.33:  # runtime can't be sort
        #     return torch.randint(0, graph.num_config.item(), (num_sample_configs,))
        
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
        
        resample_idx = torch.randint(0, len(self.intervals[ind]), [num_sample_configs])
        sample_idx = self.intervals[ind][resample_idx]
        graph.y[sample_idx]
        return sample_idx


class TPUGraphsNpz(Dataset):

    KEYS = [
        'num_config_idx', 'pestat_RWSE', 'edge_index', 'config_feats', 
        'num_config', 'eigvecs_sn', 'config_idx', 'op_feats', 'y', 
        'num_nodes', 'partptr', 'partition_idx', 'op_code', 'eigvals_sn', 'graph_name', 'source_dataset',
    ]
    
    def __init__(
          self, 
          root: str, 
          thres: int = 1000,
          transform: Optional[Callable] = None,
          pre_transform: Optional[Callable] = None,
          pre_filter: Optional[Callable] = None,
          source: str = 'nlp',  # 'nlp' or 'xla'
          search: str = 'random',  # 'random' or 'default'
          cache_in_memory: bool = False,
        ):
        assert source in ('nlp', 'xla')
        assert search in ('random', 'default')
        print(f'[TPUGraphsNpz] source: {source}, search: {search}, cache_in_memory: {cache_in_memory}')
        self.thres = thres
        self.source = source
        self.search = search
        self.epoch_multiply = 1
        self.cache_in_memory = cache_in_memory
        self._cache = {}
        super().__init__(root, transform, pre_transform, pre_filter)
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
            
            total_nodes = 0
            total_unq_segs = 0
            total_segs = 0
            total_graphs = 0
            max_nodes = 0
            
            for path in tqdm(self.processed_paths):
                data = torch.load(path)
                if isinstance(data, Data):
                    op_feats.append(data.op_feats)
                    num_node = data.op_feats.size(0)
                    num_cfgs = data.config_feats.size(0)
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
                num_nodes=total_nodes,
                num_segments=total_segs,
                num_unique_segments=total_unq_segs,
                max_node_per_graph=max_nodes,
            )
            print(self._meta)
        # self.data.op_feats = (self.data.op_feats - op_feats_mean) / op_feats_std
        return self._meta
        
    @property
    def raw_file_names(self):
        pattern = osp.join(self.raw_dir, f"npz/layout/{self.source}/{self.search}/**", '*.npz')
        raw_dir = self.raw_dir
        if not raw_dir.endswith(osp.sep):
            raw_dir += osp.sep
        relative_paths = [p.replace(raw_dir, "") for p in glob.glob(pattern, recursive=True)]
        return relative_paths

    @property
    def processed_file_names(self):
        files = [f'{self.source}_{self.search}_data_{i}.pt' for i in range(len(self.raw_file_names))]
        files.append(f'{self.source}_{self.search}_split_dict.pt')
        return files

    # def download(self):
    #     # Download to `self.raw_dir`.
    #     path = download_url(url, self.raw_dir)

    def process(self):
        split_names = ['train', 'valid', 'test']
        split_dict = {'train': [], 'valid': [], 'test': []}
        parts_cnt = 0
        
        for idx, raw_path in enumerate(tqdm(self.raw_paths)):
            out_path = osp.join(self.processed_dir, f'{self.source}_{self.search}_data_{idx}.pt')
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
            
            edge_index = torch.tensor(np_file["edge_index"].T)
            runtime = torch.tensor(np_file["config_runtime"])
            op = torch.tensor(np_file["node_feat"])
            op_code = torch.tensor(np_file["node_opcode"])
            config_feats = torch.tensor(np_file["node_config_feat"])
            config_feats = config_feats.view(-1, config_feats.shape[-1])
            config_idx = torch.tensor(np_file["node_config_ids"])  # node-indies of configurable nodes
            num_config = torch.tensor(np_file["node_config_feat"].shape[0])
            num_config_idx = torch.tensor(np_file["node_config_feat"].shape[1])  # number of configurable nodes

            if -2**7 <= config_feats.min() and config_feats.min() <= 2**7:
                config_feats = config_feats.to(torch.int8)
            
            num_nodes = torch.tensor(np_file["node_feat"].shape[0])
            num_parts = num_nodes // self.thres + 1
            interval = num_nodes // num_parts
            partptr = torch.arange(0, num_nodes, interval+1)  # TODO: Find a better way to partition graph according to topologic 
            if partptr[-1] != num_nodes:
                partptr = torch.cat([partptr, torch.tensor([num_nodes])])
            graph_name = osp.basename(raw_path).replace('.npz', '')
            
            data = Data(edge_index=edge_index, op_feats=op, op_code=op_code, 
                        config_feats=config_feats, config_idx=config_idx,
                        num_config=num_config, num_config_idx=num_config_idx, y=runtime, 
                        num_nodes=num_nodes, partptr=partptr, partition_idx=parts_cnt, 
                        graph_name=graph_name,)
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            parts_cnt += num_parts * num_config
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
            pt_file = osp.join(self.processed_dir, f'{self.source}_{self.search}_data_{idx}.pt')
            # print(f"[{getattr(self, 'split_name', '?')}]Load {pt_file}, {len(self._cache)}")
            data = torch.load(pt_file)
            if isinstance(data.partition_idx, int):  # HACK: habdle the case that PyGemo not able to convert int to tensor
                data.partition_idx = torch.tensor(data.partition_idx)
            op_feats_mean = self.meta.op_feat_mean
            op_feats_std = self.meta.op_feat_std
            data.op_feats = (data.op_feats - op_feats_mean) / op_feats_std
            
            if self.cache_in_memory:
                self._cache[idx] = copy.deepcopy(data)
        data.config_feats = data.config_feats.float()
        data.source_dataset = f"{self.source}-{self.search}-{idx}"
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
        self.custom_split_names = ['train'] + [f'valid_{v}' for v in self.dataset_names] + ['test']  # for split_generator.py
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

    def get(self, idx):
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
    
    def get_idx_split(self):
        if not hasattr(self, "_split_idxs"):
            self._split_idxs = defaultdict(list)
            start_indies = [0] + list(accumulate(len(self.datasets[k]) for k in self.dataset_names[:-1]))
            start_indies = { k: v for k, v in zip(self.dataset_names, start_indies) }
            
            for name, dataset in self.datasets.items():
                offset = start_indies[name]
                for split, idx in dataset.get_idx_split().items():
                    off_idx = [j + offset for j in idx]
                    if split == 'valid':
                        self._split_idxs[f"valid_{name}"] = off_idx
                    else:
                        self._split_idxs[split] += off_idx
        return self._split_idxs


if __name__ == '__main__':
    dataset = TPUGraphs(root='datasets/TPUGraphs')
    import pdb; pdb.set_trace()
