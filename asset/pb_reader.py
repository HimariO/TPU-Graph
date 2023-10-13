import os
import glob
from termcolor import colored
from itertools import product
from collections import defaultdict
from typing import *

import torch
import numpy as np
from tqdm import tqdm
from numba import jit, prange
from loguru import logger
import hlo_proto_py.tuning_pb2 as tuning_pb
import hlo_proto_py.hlo_pb2 as hlo_pb

"""
ref on how different arugment of hlo insturction work:
https://www.tensorflow.org/xla/operation_semantics#reshape
"""

PrimitiveTypeBits = {
  1: 1, # PRED
  21: 4, # S4
  2: 8, # S8
  3: 16, # S16
  4: 32, # S32
  5: 64, # S64

  22: 4, # U4
  6: 8, # U8
  7: 16, # U16
  8: 32, # U32
  9: 64, # U64
  10: 16, # F16
  11: 32, # F32

  16: 16, # BF16
  12: 64, # F64

  19: 8, # F8E5M2
  20: 8, # F8E4M3FN
  23: 8, # F8E4M3B11FNUZ
  24: 8, # F8E5M2FNUZ
  25: 8, # F8E4M3FNUZ

  15: 64, # C64
  18: 128, # C128
  13: -1, # TUPLE
  14: -1, # OPAQUE_TYPE
  17: -1, # TOKEN
}


# @logger.catch
@jit
def conv_read_ops(input_shape: List[int], input_layout: List[int], 
        pagesize: int=128*8, spatial_dims: Tuple[int]=(1, 2), kernel_size: Tuple[int]=(3, 3)):
    """
    Estiamte how many HBM -> register read operation will happend when we run certain conv op.
    # [B H W C] -> minor(fastest varying index) [3 2 1 0] major
    """
    
    ndim = len(input_shape)
    step_sizes = {}
    prev_dim = -1
    for j in input_layout:
        step_sizes[j] = max(1, prev_dim)
        prev_dim = input_shape[j] * max(1, prev_dim)

    last_access = -pagesize - 1
    reads = 0
    for b, c in product(range(input_shape[0]), range(input_shape[2])):
        window_stride = product(*[
            range(input_shape[si] - ks) 
            for si, ks in zip(spatial_dims, kernel_size)
        ])
        for zyx in window_stride:
            kernel_coord = product(*[range(ks) for ks in kernel_size])
            mem_loc = step_sizes[0] * b
            mem_loc += step_sizes[ndim - 1] * c
            for kis in kernel_coord:
                for dim, (si, ki) in enumerate(zip(spatial_dims, kis)):
                    mem_loc += step_sizes[si] * (zyx[dim] + ki)
            if mem_loc - last_access > pagesize or mem_loc - last_access < 0:
                reads += 1
                last_access = mem_loc - mem_loc % pagesize

    return reads


def estimate_shape(
        inst: hlo_pb.HloInstructionProto, 
        id2inst: Dict[int, hlo_pb.HloInstructionProto], 
        layout_config: Dict[int, np.ndarray]) -> List[int]:
    
    def map_shape(src_shape, layout):
        if isinstance(src_shape, torch.Tensor):
            src_shape = src_shape.tolist()
        if isinstance(layout, torch.Tensor):
            layout = layout.tolist()
        new_shape = [-1] * len(layout)
        axis = set(range(len(src_shape)))
        for i, j in enumerate(layout):
            if j >= 0:
                axis.remove(j)
                new_shape[i] = src_shape[j]
            elif axis:
                new_shape[i] = src_shape[axis.pop()]
        return new_shape

    feature = {
        'input_shape_1': [-1] * 6,
        'input_shape_2': [-1] * 6,  # kernel shape for conv op
        'output_shape': [-1] * 6,
        'input_layout_1_align': True,
        'input_layout_2_align': True,
    }
    # default_layout = list(inst.shape.layout.minor_to_major)
    config = layout_config[inst.id]
    
    if inst.opcode == "reshape":
        hidden = id2inst[inst.operand_ids[0]]
        if hidden.shape.dimensions:
            feature['input_shape_1'] = map_shape(list(hidden.shape.dimensions), config[6:12])
        else:
            feature['input_shape_1'][0] = 1  # scalar
        if hidden.id in layout_config:
            feature['input_layout_1_align'] = bool((layout_config[hidden.id][:6] == config[6:12]).all())
    elif inst.opcode in ["dot", "convolution"]:
        hidden = id2inst[inst.operand_ids[0]]  # lhs
        kernel = id2inst[inst.operand_ids[1]]  # rhs

        if hidden.shape.dimensions:
            feature['input_shape_1'] = map_shape(list(hidden.shape.dimensions), config[6:12])
        else:
            feature['input_shape_1'][0] = 1  # scalar
        if kernel.shape.dimensions:
            feature['input_shape_2'] = map_shape(list(kernel.shape.dimensions), config[12:18])
        else:
            feature['input_shape_2'][0] = 1  # scalar
        
        if hidden.id in layout_config:
            feature['input_layout_1_align'] = bool((layout_config[hidden.id][:6] == config[6:12]).all())
        if kernel.id in layout_config:
            feature['input_layout_2_align'] = bool((layout_config[kernel.id][:6] == config[6:12]).all())
        
    if inst.shape.dimensions:
        feature['output_shape'] = map_shape(list(inst.shape.dimensions), config[:6])
    else:
        feature['output_shape'][0] = 1
    
    dtype = inst.shape.element_type
    dsize = PrimitiveTypeBits[dtype]
    # for k in ['input_shape_1', 'input_shape_2', 'output_shape']:
    #     feature[k] = [v * dsize if v > 0 else v for v in feature[k]]
    flatten = []
    for k in sorted(feature.keys()):
        v = feature[k]
        if isinstance(v, list):
            flatten += v
        elif isinstance(v, bool):
            flatten += [int(v)]
        else:
            raise ValueError(f"{type(v)}, {v}")
    return flatten


def single_file_eda():
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/train/inference_mlperf_ssd_1200_batch_1.pb"
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/valid/resnet_v1_50_official_batch_128_bf16.pb"
    pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/nlp/default/valid/bert_multi_cased_L-12_H-768_A-12_batch_size_16_train.pb"
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/valid/unet_3d.4x4.bf16.pb"

    with open(pb_path, mode='rb') as f:
        hlo_obj = tuning_pb.ModuleTuningData()
        hlo_obj.ParseFromString(f.read())

        m = len(hlo_obj.module.computations)
        computes = hlo_obj.module.computations
        print(hlo_obj.module.name)
        print(m, computes[0])
        print('-' * 100)
        id2inst = {}
        inst_chds = defaultdict(list)

        for i in range(m):
            comp: hlo_pb.HloComputationProto = computes[i]
            for inst in comp.instructions:
                id2inst[inst.id] = inst
                for chd in inst.operand_ids:
                    inst_chds[chd].append(inst.id)

        tunable = [
            'dot',
            'reshape',
            # 'convolution'
        ]
        for i in range(m):
            comp: hlo_pb.HloComputationProto = computes[i]
            print(colored('*', color='green'), comp.name)
            # print(colored('*', color='green'), comp.program_shape)
            for inst in comp.instructions:
                if inst.opcode in tunable:
                    print(colored('>>  ==  --', color='yellow'))
                    print(inst)
                    print("inst.outfeed_shape: ", inst.outfeed_shape, inst.shape.element_type)
                    # for k, p in enumerate(inst_chds[inst.id]):
                    for k, p in enumerate(inst.operand_ids):
                        print(colored(f"P[{k}]", color='cyan'), id2inst[p])
        
        print(colored('>>  ==  --', color='green'))
        print(hlo_obj.runs)
        print(hlo_obj.config_index_to_node)


def create_dataset_feature(pt_glob, pb_dir):
    data_src = defaultdict(dict)
    pt_files = glob.glob(pt_glob, recursive=True)
    for path in pt_files:
        graph_name = torch.load(path).graph_name
        data_src[graph_name]['pt'] = path
    pb_files = glob.glob(os.path.join(pb_dir, "**", "*.pb"), recursive=True)
    for path in pb_files:
        graph_name = os.path.basename(path).replace(".pb", "")
        data_src[graph_name]['pb'] = path
    
    for graph_name, file_dict in tqdm(data_src.items()):
        pt_data = torch.load(file_dict['pt'])
        
        with open(file_dict['pb'], mode='rb') as f:
            hlo_obj = tuning_pb.ModuleTuningData()
            hlo_obj.ParseFromString(f.read())
            
            m = len(hlo_obj.module.computations)
            computes = hlo_obj.module.computations
            configure_nodes = hlo_obj.config_index_to_node
            row2inst = {}
            id2inst = {}
            inst_list = []  # NOTE: according to host, this follow topologit order.
            for i in range(m):
                comp: hlo_pb.HloComputationProto = computes[i]
                for inst in comp.instructions:
                    row2inst[len(inst_list)] = inst.id
                    id2inst[inst.id] = inst
                    inst_list.append(inst)
        
        all_graph_cfgs = []
        config_feats = pt_data['config_feats'].view(
            pt_data['num_config'], -1, pt_data['config_feats'].size(-1))
        for graph_config in config_feats:
            node2cfg = {
                row2inst[node]: feat 
                for node, feat in zip(pt_data['config_idx'].tolist(), graph_config)
            }
            per_graph_config = []
            for node in pt_data['config_idx'].tolist():
                shape_feat = estimate_shape(inst_list[node], id2inst, node2cfg)
                per_graph_config.append(shape_feat)
            all_graph_cfgs.append(per_graph_config)
        
        all_graph_cfgs = torch.tensor(all_graph_cfgs)
        if -2**7 < all_graph_cfgs.min() and all_graph_cfgs.max() < 2**7:
            all_graph_cfgs = all_graph_cfgs.to(torch.int8)
        elif -2**15 < all_graph_cfgs.min() and all_graph_cfgs.max() < 2**15:
            all_graph_cfgs = all_graph_cfgs.to(torch.int16)
        elif -2**31 < all_graph_cfgs.min() and all_graph_cfgs.max() < 2**31:
            all_graph_cfgs = all_graph_cfgs.to(torch.int32)
        pt_data['extra_feat'] = all_graph_cfgs
        # torch.save(file_dict['pt'])
        
            

if __name__ == '__main__':
    # single_file_eda()
    create_dataset_feature(
        "/home/ron/Projects/TPU-Graph/datasets/TPUGraphsNpz/processed/xla_default*data*.pt",
        "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/"
    )
    # print(conv_read_ops([2,24,24,24,512], [4,3,2,1,0], spatial_dims=[1,2,3], kernel_size=[2,2,2]))
    # print(conv_read_ops([2,24,24,24,512], [3,2,1,4,0], spatial_dims=[1,2,3], kernel_size=[2,2,2]))