import os
import glob
import warnings
import multiprocessing as mp
from functools import lru_cache
from termcolor import colored
from itertools import product
from collections import defaultdict
from typing import *

import torch
import numpy as np
import pysnooper
from tqdm import tqdm
from numba import jit, prange, NumbaPendingDeprecationWarning
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


class ReadEstimatorV1:
    """
    Estiamting number of read operation will happending while ignore the register padding behaivor.
    """
    
    @lru_cache(None)
    @staticmethod
    @jit(parallel=False)
    def reshape_read_ops(input_shape: List[int], input_layout: List[int], dim_permut: List[int]=[0], pagesize: int=128*8):
        # NOTE: actual output shape don't effect the performance, only dimension permuation does
        ndim = len(input_shape)
        nele = 1
        step_sizes = {}
        prev_dim = -1
        for j in input_layout:  # minor to major
            step_sizes[j] = max(1, prev_dim)
            prev_dim = input_shape[j] * max(1, prev_dim)
            nele *= input_shape[j]
        
        # output_laytout = [input_layout[i] for i in dim_permut]
        access_pattern = [1]
        for j in dim_permut[:-1]:
            access_pattern.append(access_pattern[-1] * input_shape[j])
        # access_pattern = access_pattern[::-1]
        # print('input_shape', input_shape)
        # print('access_pattern', access_pattern)

        last_access = -pagesize - 1
        reads = 0
        for i in prange(nele):
            coord = [0] * ndim  # major to minor
            iq = i
            for dim, j in zip(dim_permut, access_pattern):
                coord[dim] = (iq // j) % input_shape[dim]
            # print(coord)
            mem_loc = 0
            for dim, j in enumerate(coord):
                mem_loc += step_sizes[dim] * j
            if mem_loc - last_access > pagesize or mem_loc - last_access < 0:
                reads += 1
                last_access = mem_loc - mem_loc % pagesize
        return reads

    @lru_cache(None)
    @staticmethod
    @jit(parallel=False)
    def dot_read_ops(input_shape: Tuple[int], input_layout: Tuple[int], pagesize: int=128*8, reduce_dims=[0], fast=False):
        """
        ref: https://www.tensorflow.org/xla/operation_semantics#dotgeneral
        """
        ndim = len(input_shape)
        step_sizes = {}
        prev_dim = -1
        for j in input_layout:  # minor to major
            step_sizes[j] = max(1, prev_dim)
            prev_dim = input_shape[j] * max(1, prev_dim)
        
        vec_size = 1
        for i in reduce_dims:
            vec_size *= input_shape[i]
        prev_dim = -1
        vec_step = {}
        for j in reduce_dims[::-1]:
            vec_step[j] = max(1, prev_dim)
            prev_dim = input_shape[j] * max(1, prev_dim)
        
        batch_size = 1
        batch_dims = []
        for i in range(ndim):
            if i not in reduce_dims:
                batch_size *= input_shape[i]
                batch_dims.append(i)
        prev_dim = -1
        batch_step = {}
        for j in batch_dims[::-1]:
            batch_step[j] = max(1, prev_dim)
            prev_dim = input_shape[j] * max(1, prev_dim)
        
        last_access = -pagesize - 1
        reads = 0
        mul = max(1, batch_size // 2) if fast else 1
        for b in prange(2 if fast else batch_size):
            bq = b
            # restore the batch dims's coordinates
            batch_coord = []
            for j in batch_dims:
                batch_coord.append(bq // batch_step[j])
                bq %= batch_step[j]
            
            for c in prange(vec_size):
                cq = c
                vec_coord = []
                for j in reduce_dims:
                    vec_coord.append(cq // vec_step[j])
                    cq %= vec_step[j]

                mem_loc = 0
                for bi, j in zip(batch_coord, batch_dims):
                    mem_loc += bi * step_sizes[j]
                for vi, j in zip(vec_coord, reduce_dims):
                    mem_loc += vi * step_sizes[j]
                
                if mem_loc - last_access > pagesize or mem_loc - last_access < 0:
                    reads += 1
                    last_access = mem_loc - mem_loc % pagesize
        return reads * mul

    @staticmethod
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
        for b, c in product(range(input_shape[0]), range(input_shape[-1])):
            window_stride = product(*[
                range(input_shape[si] - ks) 
                for si, ks in zip(spatial_dims, kernel_size)
            ])
            for zyx in window_stride:
                kernel_coord = product(*[range(ks) for ks in kernel_size])
                for kis in kernel_coord:
                    mem_loc = step_sizes[0] * b
                    mem_loc += step_sizes[ndim - 1] * c
                    for dim, (si, ki) in enumerate(zip(spatial_dims, kis)):
                        mem_loc += step_sizes[si] * (zyx[dim] + ki)
                    if mem_loc - last_access > pagesize or mem_loc - last_access < 0:
                        reads += 1
                        last_access = mem_loc - mem_loc % pagesize

        return reads

    @lru_cache(None)    
    @staticmethod
    @jit(parallel=False)
    def conv_3d_read_ops(
            input_shape: Tuple[int], 
            input_layout: Tuple[int], 
            pagesize: int=128*8, 
            spatial_dims: Tuple[int]=(1, 2, 3), 
            kernel_size: Tuple[int]=(2, 2, 2),
            fast=False,
        ):
        
        ndim = len(input_shape)
        bc_dims = [i for i in range(ndim) if i not in spatial_dims]
        step_sizes = {}
        prev_dim = -1
        for j in input_layout:
            step_sizes[j] = max(1, prev_dim)
            prev_dim = input_shape[j] * max(1, prev_dim)

        last_access = -pagesize - 1
        reads = 0
        batch_size = 2 if fast else input_shape[bc_dims[0]]
        mul = max(1, batch_size // 2) if fast else 1
        for b in prange(batch_size):
            for c in prange(input_shape[bc_dims[1]]):
                
                for z in prange(input_shape[spatial_dims[0]] - kernel_size[0]):
                    for y in prange(input_shape[spatial_dims[1]] - kernel_size[1]):
                        for x in prange(input_shape[spatial_dims[2]] - kernel_size[2]):
                            
                            for k0 in prange(kernel_size[0]):
                                for k1 in prange(kernel_size[1]):
                                    for k2 in prange(kernel_size[2]):
                                        mem_loc = step_sizes[0] * b
                                        mem_loc += step_sizes[ndim - 1] * c
                                        mem_loc += step_sizes[spatial_dims[0]] * (z + k0)
                                        mem_loc += step_sizes[spatial_dims[1]] * (y + k1)
                                        mem_loc += step_sizes[spatial_dims[2]] * (x + k2)
                                        if mem_loc - last_access > pagesize or mem_loc - last_access < 0:
                                            reads += 1
                                            last_access = mem_loc - mem_loc % pagesize

        return reads * mul
    
    @lru_cache(None)
    @staticmethod
    @jit(parallel=False)
    def conv_2d_read_ops(
            input_shape: Tuple[int], 
            input_layout: Tuple[int], 
            pagesize: int=128*8, 
            spatial_dims: Tuple[int]=(1, 2), 
            kernel_size: Tuple[int]=(3, 3),
            fast=False
        ):
        ndim = len(input_shape)
        bc_dims = [i for i in range(ndim) if i not in spatial_dims]
        step_sizes = {}
        prev_dim = -1
        for j in input_layout:
            step_sizes[j] = max(1, prev_dim)
            prev_dim = input_shape[j] * max(1, prev_dim)

        last_access = -pagesize - 1
        reads = 0
        batch_size = 2 if fast else input_shape[bc_dims[0]]
        mul = max(1, batch_size // 2) if fast else 1
        for b in prange(batch_size):
            for c in prange(input_shape[bc_dims[1]]):
                
                for z in prange(input_shape[spatial_dims[0]] - kernel_size[0]):
                    for y in prange(input_shape[spatial_dims[1]] - kernel_size[1]):
                        
                        for k0 in prange(kernel_size[0]):
                            for k1 in prange(kernel_size[1]):
                                mem_loc = step_sizes[0] * b
                                mem_loc += step_sizes[ndim - 1] * c
                                mem_loc += step_sizes[spatial_dims[0]] * (z + k0)
                                mem_loc += step_sizes[spatial_dims[1]] * (y + k1)
                                if mem_loc - last_access > pagesize or mem_loc - last_access < 0:
                                    reads += 1
                                    last_access = mem_loc - mem_loc % pagesize

        return reads * mul

    @staticmethod
    def esitmate(
            inst: hlo_pb.HloInstructionProto,
            id2inst: Dict[int, hlo_pb.HloInstructionProto], 
            layout_config: Dict[int, np.ndarray]) -> List[int]:
        features = {
            'in1_reads': 0,
            'in2_reads': 0,
        }
        output_shape = tuple(inst.shape.dimensions)
        config = layout_config[inst.id].int().tolist()
        ndim = len(output_shape)
        
        def cfg2layout(cfg, ndim):
            dims = list(range(ndim))
            layout = []
            for i in cfg:
                if i > -1:
                    layout.append(i)
                    dims.remove(i)
                elif dims:
                    layout.append(dims.pop())
            return tuple(layout)
        
        out_layout = cfg2layout(config[:6], ndim)
        out_layout = tuple(out_layout)
        
        input_1 = id2inst[inst.operand_ids[0]]
        input_shape_1 = tuple(input_1.shape.dimensions)
        in1_layout = cfg2layout(config[6:12], len(input_shape_1))
        
        if inst.opcode == "reshape":
            features['in1_reads'] = ReadEstimatorV1.reshape_read_ops(input_shape_1, in1_layout, dim_permut=out_layout)
        else:
            input_2 = id2inst[inst.operand_ids[1]]
            input_shape_2 = tuple(input_2.shape.dimensions)
            in2_layout = cfg2layout(config[12:], len(input_shape_2))
            
            if inst.opcode == 'convolution':
                kernel_size = tuple([input_shape_2[i] for i in inst.convolution_dimension_numbers.kernel_spatial_dimensions])
                spatial_dims = tuple(inst.convolution_dimension_numbers.input_spatial_dimensions)
                kernel_spatial_dims = tuple(inst.convolution_dimension_numbers.kernel_spatial_dimensions)
                
                if len(input_shape_1) == 4:
                    features['in1_reads'] = ReadEstimatorV1.conv_2d_read_ops(
                        input_shape_1, 
                        in1_layout, 
                        spatial_dims=spatial_dims, 
                        kernel_size=kernel_size,
                        fast=True
                    )
                elif len(input_shape_1) == 5:
                    features['in1_reads'] = ReadEstimatorV1.conv_3d_read_ops(
                        input_shape_1, 
                        in1_layout, 
                        spatial_dims=spatial_dims, 
                        kernel_size=kernel_size,
                        fast=True
                    )
                else:
                    # TODO: maybe add support for Conv1D?
                    raise ValueError(f"Encoutner unsupported {len(input_shape_1) - 2} dimension conv operation")
                
                if len(input_shape_2) == 4:
                    features['in2_reads'] = ReadEstimatorV1.conv_2d_read_ops(
                        input_shape_2, 
                        in2_layout, 
                        spatial_dims=kernel_spatial_dims, 
                        kernel_size=kernel_size,
                        fast=True
                    )
                elif len(input_shape_2) == 5:
                    features['in2_reads'] = ReadEstimatorV1.conv_3d_read_ops(
                        input_shape_2, 
                        in2_layout, 
                        spatial_dims=kernel_spatial_dims, 
                        kernel_size=kernel_size,
                        fast=True
                    )
                else:
                    raise ValueError(f"Encoutner unsupported {len(input_shape_2) - 2} dimension conv operation")
            elif inst.opcode == 'dot':
                reduce_dims = tuple(inst.dot_dimension_numbers.lhs_contracting_dimensions)
                features['in1_reads'] = ReadEstimatorV1.dot_read_ops(input_shape_1, in1_layout, reduce_dims=reduce_dims, fast=True)
                reduce_dims = tuple(inst.dot_dimension_numbers.rhs_contracting_dimensions)
                features['in2_reads'] = ReadEstimatorV1.dot_read_ops(input_shape_2, in2_layout, reduce_dims=reduce_dims, fast=True)
            
            return [
                max(1, features['in1_reads'] // 100) if features['in1_reads'] > 0 else 0,
                max(1, features['in2_reads'] // 100) if features['in2_reads'] > 0 else 0,
            ]


def estimate_shape(
        inst: hlo_pb.HloInstructionProto, 
        id2inst: Dict[int, hlo_pb.HloInstructionProto], 
        layout_config: Dict[int, np.ndarray]) -> List[int]:
    
    def apply_layout(src_shape, layout):
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
            feature['input_shape_1'] = apply_layout(list(hidden.shape.dimensions), config[6:12])
        else:
            feature['input_shape_1'][0] = 1  # scalar
        if hidden.id in layout_config:
            feature['input_layout_1_align'] = bool((layout_config[hidden.id][:6] == config[6:12]).all()  or (config[6:12] == -1).all())
    elif inst.opcode in ["dot", "convolution"]:
        hidden = id2inst[inst.operand_ids[0]]  # lhs
        kernel = id2inst[inst.operand_ids[1]]  # rhs

        if hidden.shape.dimensions:
            feature['input_shape_1'] = apply_layout(list(hidden.shape.dimensions), config[6:12])
        else:
            feature['input_shape_1'][0] = 1  # scalar
        if kernel.shape.dimensions:
            feature['input_shape_2'] = apply_layout(list(kernel.shape.dimensions), config[12:18])
        else:
            feature['input_shape_2'][0] = 1  # scalar
        
        if hidden.id in layout_config:
            input_layout = layout_config[hidden.id][:6]
            feature['input_layout_1_align'] = bool((input_layout == config[6:12]).all() or (config[6:12] == -1).all())
        if kernel.id in layout_config:
            input_layout = layout_config[kernel.id][:6]
            feature['input_layout_2_align'] = bool((layout_config[kernel.id][:6] == config[12:18]).all() or (config[12:18] == -1).all())
        
    if inst.shape.dimensions:
        feature['output_shape'] = apply_layout(list(inst.shape.dimensions), config[:6])
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
    # if not (feature['input_layout_1_align'] and feature['input_layout_2_align']):
    #     logger.info(f"mis-align! {[id2inst[j].opcode for j in inst.operand_ids]} -> {inst.opcode}")
    return flatten


def single_file_eda():
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/train/inference_mlperf_ssd_1200_batch_1.pb"
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/valid/resnet_v1_50_official_batch_128_bf16.pb"
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/nlp/default/valid/bert_multi_cased_L-12_H-768_A-12_batch_size_16_train.pb"
    pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/nlp/default/valid/albert_en_xlarge_batch_size_16_test.pb"
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
            # 'reshape',
            'convolution'
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


# @pysnooper.snoop()
def process_graph(data_src):
    bar = tqdm(data_src.items()) if len(data_src) > 1 else data_src.items()
    for graph_name, file_dict in bar:
        pt_data = torch.load(file_dict['pt'])
        
        with open(file_dict['pb'], mode='rb') as f:
            hlo_obj = tuning_pb.ModuleTuningData()
            hlo_obj.ParseFromString(f.read())
            
            m = len(hlo_obj.module.computations)
            computes = hlo_obj.module.computations
            row2inst = {}
            id2inst = {}
            inst_list = []  # NOTE: according to host, this follow topologit order.
            for i in range(m):
                comp: hlo_pb.HloComputationProto = computes[i]
                for inst in comp.instructions:
                    row2inst[len(inst_list)] = inst.id
                    id2inst[inst.id] = inst
                    inst_list.append(inst)
        
        def signature(inst, node2cfg):
            sig_arr = [inst.id, tuple(node2cfg[inst.id].tolist())]
            for i in range(2):
                if i >= len(inst.operand_ids):
                    sig_arr.append(-1)
                    sig_arr.append(-1)
                else:
                    sig_arr.append(inst.operand_ids[i])
                    if inst.operand_ids[i] in node2cfg:
                        sig_arr.append(tuple(node2cfg[inst.operand_ids[i]].tolist()))
                    else:
                        sig_arr.append(-1)
            return tuple(sig_arr)
        
        cache_featrue = {}
        all_graph_cfgs = []
        all_graph_opreads = []
        config_feats = pt_data['config_feats'].view(
            pt_data['num_config'], -1, pt_data['config_feats'].size(-1))
        
        for graph_config in config_feats:
            node2cfg = {
                row2inst[node]: feat 
                for node, feat in zip(pt_data['config_idx'].tolist(), graph_config)
            }
            per_graph_config = []
            per_graph_reads = []
            for node in pt_data['config_idx'].tolist():
                sig = signature(inst_list[node], node2cfg)
                if sig in cache_featrue:
                    shape_feat = cache_featrue[sig]
                else:
                    shape_feat = estimate_shape(inst_list[node], id2inst, node2cfg)
                    cache_featrue[sig] = shape_feat
                per_graph_config.append(shape_feat)
                
                reads_feat = ReadEstimatorV1.esitmate(inst_list[node], id2inst, node2cfg)
                per_graph_reads.append(reads_feat)
            all_graph_cfgs.append(per_graph_config)
            all_graph_opreads.append(per_graph_reads)
        
        def auto_dtype(ten):
            if -2**7 < ten.min() and ten.max() < 2**7:
                ten = ten.to(torch.int8)
            elif -2**15 < ten.min() and ten.max() < 2**15:
                ten = ten.to(torch.int16)
            elif -2**31 < ten.min() and ten.max() < 2**31:
                ten = ten.to(torch.int32)
            return ten
        
        all_graph_cfgs = torch.tensor(all_graph_cfgs)
        all_graph_opreads = torch.tensor(all_graph_opreads)
        pt_data['extra_feat'] = auto_dtype(all_graph_cfgs)
        pt_data['extra_read_ops_feat'] = auto_dtype(all_graph_opreads)
        torch.save(pt_data, file_dict['pt'])


@logger.catch(reraise=True)
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
    
    process_graph(data_src)
    # with mp.Pool(processes=16) as pool:
    #     breakdown = [{k: v} for k, v in data_src.items()]
    #     done = 0
    #     for _ in pool.imap_unordered(process_graph, breakdown):
    #         done += 1
    #         logger.info(f"{done}/{len(breakdown)}")
            

def fast_mode_sanity_check():
    # ratios = []
    # for batch in [1, 2, 4, 8]:
    #     for channel in [1, 4, 8, 16, 32]:
    #         layout_1 = ReadEstimatorV1.conv_3d_read_ops([1,24,24,24,channel], [4,3,2,1,0], spatial_dims=[1,2,3], kernel_size=[2,2,2])
    #         layout_2 = ReadEstimatorV1.conv_3d_read_ops([1,24,24,24,channel], [3,2,1,4,0], spatial_dims=[1,2,3], kernel_size=[2,2,2])
    #         print(f"b = {batch}, c = {channel}, {layout_1} / {layout_2} = {layout_1 / layout_2}")
    #         ratios.append(layout_1 / layout_2)
    # print(ratios)

    ratios = []
    fast_mode = False
    for b1, b2 in product([1, 2, 4, 8], [1, 2, 4, 8]):
        layout_1 = ReadEstimatorV1.dot_read_ops((b1, b2, 32, 64), (0, 1, 2, 3), reduce_dims=(2,3), fast=fast_mode)
        layout_2 = ReadEstimatorV1.dot_read_ops((b1, b2, 32, 64), (2, 3, 0, 1), reduce_dims=(2,3), fast=fast_mode)
        print(f"b = {[b1, b2]}, {layout_1} / {layout_2} = {layout_1 / layout_2}")
        ratios.append(layout_1 / layout_2)

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # single_file_eda()
    # create_dataset_feature(
    #     "/home/ron_zhu/TPU-Graph/datasets/TPUGraphsNpz/processed/xla_default*data*.pt",
    #     "/home/ron_zhu/data/tpugraphs/pb/pb/layout/xla/default/"
    # )
    create_dataset_feature(
        "/home/ron/Projects/TPU-Graph/datasets/TPUGraphsNpz/processed/xla_default*data*.pt",
        "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/"
    )

    # print(ReadEstimatorV1.conv_read_ops([2,24,24,24,64], [4,3,2,1,0], spatial_dims=[1,2,3], kernel_size=[2,2,2]))
    # print(ReadEstimatorV1.conv_read_ops([2,24,24,24,64], [3,2,1,4,0], spatial_dims=[1,2,3], kernel_size=[2,2,2]))
    # print(ReadEstimatorV1.conv_3d_read_ops([2,24,24,24,64], [4,3,2,1,0], spatial_dims=[1,2,3], kernel_size=[2,2,2]))
    # print(ReadEstimatorV1.conv_3d_read_ops([2,24,24,24,64], [3,2,1,4,0], spatial_dims=[1,2,3], kernel_size=[2,2,2]))
    print(ReadEstimatorV1.conv_2d_read_ops((2,24,24,64), (3,2,1,0), spatial_dims=(1,2), kernel_size=(2,2)))
    
    # print(ReadEstimatorV1.dot_read_ops([23, 64, 2048], [2, 1, 0], reduce_dims=[1,0]))
    # print(ReadEstimatorV1.dot_read_ops([23, 64, 2048], [1, 0, 2], reduce_dims=[1,0]))
    # print(ReadEstimatorV1.dot_read_ops([23, 64, 2048], [0, 1, 2], reduce_dims=[1,0]))
    # fast_mode_sanity_check()

    # print(ReadEstimatorV1.reshape_read_ops([4,2,3], [2,1,0], [0,2,1]))
    # print(ReadEstimatorV1.reshape_read_ops((40,20,30), (2,1,0), (0,2,1)))