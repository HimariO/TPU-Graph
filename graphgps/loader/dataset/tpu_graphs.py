from typing import Optional, Callable, List
import copy
import re
import os
import glob
import os.path as osp
import numpy as np
import torch
from torch_geometric.data import (InMemoryDataset, Data, download_url,
                                  extract_tar, extract_zip)
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
                    partptr = torch.arange(0, num_nodes, interval+1)
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

if __name__ == '__main__':
    dataset = TPUGraphs(root='datasets/TPUGraphs')
    import pdb; pdb.set_trace()
