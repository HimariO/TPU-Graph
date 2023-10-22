import os
import glob
import math
import warnings
import multiprocessing as mp
from functools import lru_cache
from termcolor import colored
from itertools import product, zip_longest
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


def eval_opa(y_true, y_pred):
    num_preds = y_pred.shape[0]
    i_idx = torch.arange(num_preds).repeat(num_preds)
    j_idx = torch.arange(num_preds).repeat_interleave(num_preds)
    pairwise_true = y_true[i_idx] > y_true[j_idx]
    opa_indices = pairwise_true.nonzero()[0].flatten()
    opa_preds = y_pred[i_idx[opa_indices]] - y_pred[j_idx[opa_indices]]
    if len(opa_indices) > 0:
        opa_acc = float((opa_preds > 0).sum()) / opa_preds.shape[0]
    else:
        opa_acc = 0.0
    return opa_acc


class ReadEstimatorV1:
    """
    Estiamting number of read operation will happending while ignore the register padding behaivor.
    """
    
    @lru_cache(None)
    @staticmethod
    def reshape_read_ops(
            input_shape: Tuple[int],
            input_layout: Tuple[int],
            output_shape: Tuple[int],
            dim_permut: Tuple[int]=[0],
            pagesize: int=128*8,
            fast=False) -> int:
        # NOTE: actual output shape don't effect the performance, only dimension permuation does
        ndim_in = len(input_shape)
        ndim_out = len(output_shape)
        nele = 1
        
        step_sizes = {}
        prev_dim = -1
        for j in input_layout:  # minor to major
            step_sizes[j] = float(max(1, prev_dim))
            prev_dim = input_shape[j] * max(1, prev_dim)
            nele *= input_shape[j]
        
        if nele < pagesize:
            return 1
        
        out_step_sizes = {}
        prev_dim = -1
        for j in range(ndim_out - 1, -1, -1):  # minor to major
            out_step_sizes[j] = float(max(1, prev_dim))
            prev_dim = output_shape[j] * max(1, prev_dim)
        
        # output_laytout = [input_layout[i] for i in dim_permut]
        access_pattern = [1]
        for j in dim_permut[:-1]:
            access_pattern.append(access_pattern[-1] * output_shape[j])
        in_acc_pattern = [1]
        for j in input_shape[::-1][:-1]:
            in_acc_pattern.append(in_acc_pattern[-1] * j)
        
        mul = 1
        if fast:
            argmax = input_shape.index(max(input_shape))
            ratio = max(1, input_shape[argmax] // 32)
            mul *= ratio
            nele //= ratio

        last_access = -pagesize - 1
        reads = 0
        for i in prange(nele):
            dst_coord = [0] * ndim_out  # major to minor
            iq = i
            for dim, j in zip(dim_permut, access_pattern):
                dst_coord[dim] = (iq // j) % output_shape[dim]
            # HACK: we have to compute dst_1d in this way to avoid numba compile error
            dst_1d = 0
            for a, b in enumerate(dst_coord):
                dst_1d += out_step_sizes[a] * b
            dst_1d = int(dst_1d)
            # print(i, dst_coord, dst_1d)
            
            src_coord = [0] * ndim_in
            for dim, j in enumerate(in_acc_pattern[::-1]):
                src_coord[dim] = dst_1d // j
                dst_1d %= j
            # print(src_coord)
            mem_loc = 0
            for dim, j in enumerate(src_coord):
                mem_loc += step_sizes[dim] * j
            if mem_loc - last_access > pagesize or mem_loc - last_access < 0:
                reads += 1
                last_access = mem_loc - int(mem_loc) % pagesize
        return reads * int(mul)

    @lru_cache(None)
    @staticmethod
    def dot_read_ops(input_shape: Tuple[int], input_layout: Tuple[int], pagesize: int=128*8, reduce_dims=[0], fast=False):
        """
        ref: https://www.tensorflow.org/xla/operation_semantics#dotgeneral
        """
        ndim = len(input_shape)
        step_sizes = {}
        prev_dim = -1
        for j in input_layout:  # minor to major
            step_sizes[j] = float(max(1, prev_dim))
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
                
                if (mem_loc - last_access > pagesize) or (mem_loc - last_access < 0):
                    reads += 1
                    last_access = mem_loc - (int(mem_loc) % pagesize)
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
    def conv_3d_read_ops(
            input_shape: Tuple[int], 
            input_layout: Tuple[int], 
            pagesize: int=128*8, 
            spatial_dims: Tuple[int]=(1, 2, 3), 
            kernel_size: Tuple[int]=(2, 2, 2),
            fast=False,
        ) -> int:
        
        ndim = len(input_shape)
        bc_dims = [i for i in range(ndim) if i not in spatial_dims]
        step_sizes = {}
        prev_dim = -1
        for j in input_layout:
            step_sizes[j] = float(max(1, prev_dim))
            prev_dim = input_shape[j] * max(1, prev_dim)

        last_access = -pagesize - 1
        reads = 0
        batch_size = 2 if fast else input_shape[bc_dims[0]]
        mul = max(1, batch_size // 2) if fast else 1
        if fast:
            bound = max(16, max(kernel_size))
            input_shape = list(input_shape)
            for i in spatial_dims:
                if input_shape[i] > bound:
                    mul *= input_shape[i] / bound
                    input_shape[i] = bound
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
                                        if (mem_loc - last_access > pagesize) or (mem_loc - last_access < 0):
                                            reads += 1
                                            last_access = mem_loc - (int(mem_loc) % pagesize)

        return reads * int(mul)
    
    @lru_cache(None)
    @staticmethod
    def conv_2d_read_ops(
            input_shape: Tuple[int], 
            input_layout: Tuple[int], 
            pagesize: int=128*8, 
            spatial_dims: Tuple[int]=(1, 2), 
            kernel_size: Tuple[int]=(3, 3),
            fast=False
        ) -> int:
        ndim = len(input_shape)
        bc_dims = [i for i in range(ndim) if i not in spatial_dims]
        step_sizes = {}
        prev_dim = -1
        for j in input_layout:
            step_sizes[j] = float(max(1, prev_dim))
            prev_dim = input_shape[j] * max(1, prev_dim)

        batch_size = 2 if fast else input_shape[bc_dims[0]]
        mul = max(1, batch_size // 2) if fast else 1
        if fast:
            bound = max(24, max(kernel_size))
            input_shape = list(input_shape)
            for i in spatial_dims:
                if input_shape[i] > bound:
                    mul *= input_shape[i] / bound
                    input_shape[i] = bound
        
        last_access = -pagesize - 1
        reads = 0
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
                                if (mem_loc - last_access > pagesize) or (mem_loc - last_access < 0):
                                    reads += 1
                                    last_access = mem_loc - (int(mem_loc) % pagesize)

        return reads * int(mul)

    @classmethod
    def esitmate(
            cls,
            inst: hlo_pb.HloInstructionProto,
            id2inst: Dict[int, hlo_pb.HloInstructionProto], 
            layout_config: Dict[int, np.ndarray]) -> List[int]:
        features = {
            'in1_reads': 0,
            'in2_reads': 0,
        }
        config = layout_config[inst.id].int().tolist()
        
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
        
        output_shape = tuple(inst.shape.dimensions)
        ndim = len(output_shape)
        out_layout = cfg2layout(config[:6], ndim)
        out_layout = tuple(out_layout)
        
        input_1 = id2inst[inst.operand_ids[0]]
        input_shape_1 = tuple(input_1.shape.dimensions)
        in1_layout = cfg2layout(config[6:12], len(input_shape_1))
        
        if inst.opcode == "reshape":
            features['in1_reads'] = cls.reshape_read_ops(
                input_shape_1,
                in1_layout,
                output_shape,
                dim_permut=out_layout,
                fast=True,
            )
        else:
            input_2 = id2inst[inst.operand_ids[1]]
            input_shape_2 = tuple(input_2.shape.dimensions)
            in2_layout = cfg2layout(config[12:], len(input_shape_2))
            
            if inst.opcode == 'convolution':
                kernel_size = tuple([input_shape_2[i] for i in inst.convolution_dimension_numbers.kernel_spatial_dimensions])
                spatial_dims = tuple(inst.convolution_dimension_numbers.input_spatial_dimensions)
                kernel_spatial_dims = tuple(inst.convolution_dimension_numbers.kernel_spatial_dimensions)
                
                if len(input_shape_1) == 4:
                    features['in1_reads'] = cls.conv_2d_read_ops(
                        input_shape_1, 
                        in1_layout, 
                        spatial_dims=spatial_dims, 
                        kernel_size=kernel_size,
                        fast=True
                    )
                elif len(input_shape_1) == 5:
                    features['in1_reads'] = cls.conv_3d_read_ops(
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
                    features['in2_reads'] = cls.conv_2d_read_ops(
                        input_shape_2, 
                        in2_layout, 
                        spatial_dims=kernel_spatial_dims, 
                        kernel_size=kernel_size,
                        fast=True
                    )
                elif len(input_shape_2) == 5:
                    features['in2_reads'] = cls.conv_3d_read_ops(
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
                features['in1_reads'] = cls.dot_read_ops(input_shape_1, in1_layout, reduce_dims=reduce_dims, fast=True)
                reduce_dims = tuple(inst.dot_dimension_numbers.rhs_contracting_dimensions)
                features['in2_reads'] = cls.dot_read_ops(input_shape_2, in2_layout, reduce_dims=reduce_dims, fast=True)
            
        return [
            max(1, features['in1_reads'] // 100) if features['in1_reads'] > 0 else 0,
            max(1, features['in2_reads'] // 100) if features['in2_reads'] > 0 else 0,
        ]


class ReadEstimatorV2(ReadEstimatorV1):
    """
    TPU MMA(Mtx mupltiply and accumalate) cycle: https://youtu.be/ot4RWfGTtOg?t=693
    """
    
    @lru_cache(None)
    @staticmethod
    def conv_2d_read_ops(
            input_shape: Tuple[int], 
            input_layout: Tuple[int], 
            tilesize: int=(128, 8), # minor to major / VMEM shape
            spatial_dims: Tuple[int]=(1, 2), 
            kernel_size: Tuple[int]=(3, 3),
            fast=False,
        ) -> int:
        ndim = len(input_shape)
        is_kernel = all([input_shape[si] == k for si, k in zip(spatial_dims, kernel_size)])
        pagesize = min(input_shape[input_layout[0]], tilesize[0])
        if ndim > 1:
            pagesize *= min(input_shape[input_layout[1]], tilesize[1])
        
        bc_dims = [i for i in range(ndim) if i not in spatial_dims]
        step_sizes = {}
        prev_dim = -1
        for j in input_layout:
            step_sizes[j] = float(max(1, prev_dim))
            prev_dim = input_shape[j] * max(1, prev_dim)

        batch_size = 2 if fast else input_shape[bc_dims[0]]
        mul = max(1, batch_size // 2) if fast else 1
        if fast:
            bound = max(24, max(kernel_size))
            input_shape = list(input_shape)
            for i in spatial_dims:
                if input_shape[i] > bound:
                    mul *= input_shape[i] / bound
                    input_shape[i] = bound
        
        last_access = -pagesize - 1
        reads = 0
        for b in prange(batch_size):
            for z in prange(input_shape[spatial_dims[0]] - kernel_size[0]):
                for y in prange(input_shape[spatial_dims[1]] - kernel_size[1]):
                    if is_kernel:
                        for c in prange(input_shape[bc_dims[1]]):
                            mem_loc = step_sizes[0] * b
                            mem_loc += step_sizes[ndim - 1] * c
                            mem_loc += step_sizes[spatial_dims[0]] * z
                            mem_loc += step_sizes[spatial_dims[1]] * y
                            if (mem_loc - last_access > pagesize) or (mem_loc - last_access < 0):
                                reads += 1
                                last_access = mem_loc - (int(mem_loc) % pagesize)
                    else:
                        for k0 in prange(kernel_size[0]):
                            for k1 in prange(kernel_size[1]):
                                for c in prange(input_shape[bc_dims[1]]):
                                    mem_loc = step_sizes[0] * b
                                    mem_loc += step_sizes[ndim - 1] * c
                                    mem_loc += step_sizes[spatial_dims[0]] * (z + k0)
                                    mem_loc += step_sizes[spatial_dims[1]] * (y + k1)
                                    if (mem_loc - last_access > pagesize) or (mem_loc - last_access < 0):
                                        reads += 1
                                        last_access = mem_loc - (int(mem_loc) % pagesize)

        return reads * int(mul)
    

class ReadEstimatorV3(ReadEstimatorV1):

    @lru_cache(None)
    @staticmethod
    def reshape_read_ops(
            input_shape: Tuple[int],
            input_layout: Tuple[int],
            output_shape: Tuple[int],
            dim_permut: Tuple[int]=[0],
            vmem: int=(128, 8),
            fast=False) -> int:
        if dim_permut[:2] == input_layout[:2]:
            return 1
        
        src_chunks = 1
        _input_shape = [input_shape[i] for i in input_layout]
        for ten, mem in zip_longest(_input_shape, vmem, fillvalue=1):
            src_chunks *= math.ceil(ten / mem)
        
        dst_chunks = 1
        _output_shape = [output_shape[i] for i in dim_permut]
        for ten, mem in zip_longest(_output_shape, vmem, fillvalue=1):
            dst_chunks *= math.ceil(ten / mem)
        
        return src_chunks + dst_chunks
    
    @staticmethod
    def tensor_read_ops(input_shape: Tuple[int], input_layout: Tuple[int], vmem: int=(128, 8)):
        new_shape = []  # dim-n ~ dim-0, minor to major, reverse from normal shape format
        for dim in input_layout:
            new_shape.append(input_shape[dim])
        chunks = 1
        for ten, mem in zip_longest(new_shape, vmem, fillvalue=1):
            chunks *= math.ceil(ten / mem)
        return chunks
        
    @lru_cache(None)
    @staticmethod
    def conv_2d_read_ops(
            input_shape: Tuple[int], 
            input_layout: Tuple[int], 
            vmem: int=(128, 8), 
            spatial_dims: Tuple[int]=(1, 2), 
            kernel_size: Tuple[int]=(3, 3),
            fast=False,
        ) -> int:
        return ReadEstimatorV3.tensor_read_ops(input_shape, input_layout)

    @lru_cache(None)
    @staticmethod
    def conv_3d_read_ops(
            input_shape: Tuple[int], 
            input_layout: Tuple[int], 
            vmem: int=(128, 8), 
            spatial_dims: Tuple[int]=(1, 2), 
            kernel_size: Tuple[int]=(3, 3),
            fast=False,
        ) -> int:
        return ReadEstimatorV3.conv_2d_read_ops(input_shape, input_layout)
    
    @lru_cache(None)
    @staticmethod
    def dot_read_ops(input_shape: Tuple[int], input_layout: Tuple[int], pagesize: int=128*8, reduce_dims=[0], fast=False):
        vector_size = 1
        remains = []
        for dim in input_layout:
            if dim in reduce_dims:
                vector_size *= input_shape[dim]
            else:
                remains.append(input_shape[dim])
        new_shape = tuple(remains + [vector_size])
        default_layout = tuple(range(len(new_shape)))[::-1]
        prepare_ops = ReadEstimatorV3.reshape_read_ops(input_shape, input_layout, new_shape, dim_permut=default_layout)
        return prepare_ops + ReadEstimatorV3.tensor_read_ops(new_shape, default_layout)