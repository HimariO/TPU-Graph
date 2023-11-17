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
from estimator import (
    ReadEstimatorV1, 
    ReadEstimatorV3, 
    eval_opa
)

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
    pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/valid/bert_pretraining.4x4.fp16.pb"
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/train/transformer.4x4.fp16.pb"
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/train/ncf.2x2.fp32.pb"
    
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/test/05ae41e26dd3c4c06390371a0423233c.pb"  # eff-b7
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/test/5335ed13823b0a518ee3c79ba4425f34.pb"  # eff-b7
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/test/cd708819d3f5103afd6460b15e74eaf3.pb"  # MLP
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/test/e8a3a1401b5e79f66d7037e424f3b6df.pb"  # bert_classifier/sentence_prediction
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/test/3e7156ac468dfb75cf5c9615e1e5887d.pb"  # bert
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/test/937ee0eb0d5d6151b7b8252933b5c1c9.pb"  # resnet 50
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/test/db59a991b7c607634f13570d52ce885f.pb"  # conv net
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/test/fbaa8bb6a1aed9988281085c91065c05.pb"  # self_suppresion -> nms/(https://github.com/tensorflow/tpu/blob/master/models/official/detection/ops/nms.py)
        
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/nlp/default/test/016ac66a44a906a695afd2228509046a.pb"  # 8 layer bert, batch 64, hidden 128
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/nlp/default/test/171b0513d8874a427ccfa46d136fbadc.pb"  # 12 layer bert, 768
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/nlp/default/test/23559853d9702baaaacbb0c83fd32266.pb"  # 4 layer bert encoder
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/nlp/default/test/29886a50d55cfe77a9497bc906c76ce9.pb"  # 4 layer bert
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/nlp/default/test/32531d07a084b319dce484f53a4cf3fc.pb"  # 4 layer bert 512
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/nlp/default/test/38524e2ff135ded55b5286407e7af6b7.pb"  # 12 layer bert, 8-768
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/nlp/default/test/3a0c5517a87df8d82fd637b83298a3ba.pb"  # 24 layer transformer (tfm.nlp.networks.EncoderScaffold)
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/nlp/default/test/492c7a94d559aa4a88769142d2a68362.pb"
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/nlp/default/test/58cc2e418c3a8a19b871e15964b534ad.pb"
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/nlp/default/test/60880ed76de53f4d7a1b960b24f20f7d.pb"
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/nlp/default/test/6c1101f6231f4d1722c3b9f6d1e25026.pb"
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/nlp/default/test/7105451001e119f65b66570d170b94a8.pb"
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/nlp/default/test/71b79ca6db513e7979c3702c595150c2.pb"
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/nlp/default/test/7f6284ebe027b1e9a3850fc703858a59.pb"
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/nlp/default/test/b2fdde3b72980907578648774101543e.pb"
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/nlp/default/test/d15316c12eefdef1ba549eb433797f77.pb"
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/nlp/default/test/f6c146fc5cf10be4f3accbaca9897311.pb"
    
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/nlp/default/valid/bert_multi_cased_L-12_H-768_A-12_batch_size_16_train.pb"
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/nlp/default/valid/albert_en_xlarge_batch_size_16_test.pb"
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
                if inst.opcode in tunable and "gradie" not in inst.metadata.op_name:
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
def process_graph(data_src, skip=-1):
    bar = tqdm(data_src.items()) if len(data_src) > 1 else data_src.items()
    for step, (graph_name, file_dict) in enumerate(bar):
        if step < skip: 
            logger.warning(f"Skip {graph_name}")
            continue
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

        # if not (pt_data.y > 1e-6).any(): continue
        
        for ci, graph_config in enumerate(config_feats):
            node2cfg = {
                row2inst[node]: feat 
                for node, feat in zip(pt_data['config_idx'].tolist(), graph_config)
            }
            per_graph_config = []
            per_graph_reads = []
            for ni, node in enumerate(pt_data['config_idx'].tolist()):
                sig = signature(inst_list[node], node2cfg)
                if sig in cache_featrue:
                    shape_feat = cache_featrue[sig]
                else:
                    shape_feat = estimate_shape(inst_list[node], id2inst, node2cfg)
                    cache_featrue[sig] = shape_feat
                per_graph_config.append(shape_feat)
                
                reads_feat = ReadEstimatorV3.esitmate(inst_list[node], id2inst, node2cfg)
                per_graph_reads.append(reads_feat)
                if len(data_src) > 1:
                    logger.info(
                        f"{ci}/{len(config_feats)}, {ni}/{len(pt_data['config_idx'])},"
                        f" {inst_list[node].opcode} {list(inst_list[node].shape.dimensions)}"
                    )
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
        if (pt_data.y > 1e-6).any() and False:
            opreads = all_graph_opreads.sum(dim=-1)
            # min_val = opreads.min(dim=0, keepdim=True).values
            # max_val = opreads.max(dim=0, keepdim=True).values
            # opreads = (opreads - min_val) / torch.clip(max_val - min_val, min=1) # norm per node
            opreads = opreads.sum(dim=-1)
            feat_opa = eval_opa(pt_data.y, opreads)
            logger.info(f"{graph_name}: read ops <--> runtime OPA = {feat_opa}, {opreads.shape}")
        pt_data['extra_feat'] = auto_dtype(all_graph_cfgs)
        pt_data['extra_read_ops_feat'] = all_graph_opreads.float()
        torch.save(pt_data, file_dict['pt'])


@logger.catch(reraise=True)
def create_dataset_feature(pt_glob, pb_dir, debug=False):
    data_src = defaultdict(dict)
    pt_files = glob.glob(pt_glob, recursive=True)
    logger.info(f'Scanning files in source folder: {pt_glob}, {len(pt_files)}')
    for path in pt_files:
        graph_name = torch.load(path).graph_name
        data_src[graph_name]['pt'] = path
    pb_files = glob.glob(os.path.join(pb_dir, "**", "*.pb"), recursive=True)
    for path in pb_files:
        graph_name = os.path.basename(path).replace(".pb", "")
        data_src[graph_name]['pb'] = path
    
    logger.info(f'Start processing...')
    if debug:
        process_graph(data_src)
    else:
        workers = max(1, mp.cpu_count() // 2)
        logger.info(f"Launching {workers} worker processes")
        with mp.Pool(processes=workers) as pool:
            breakdown = [{k: v} for k, v in data_src.items()]
            done = 0
            for _ in pool.imap_unordered(process_graph, breakdown):
                done += 1
            logger.info(f"{done}/{len(breakdown)}")
            

def fast_mode_sanity_check():
    ratios = []
    for batch in [1, 2, 4, 8]:
        for channel in [1, 4, 8, 16, 32]:
            layout_1 = ReadEstimatorV3.conv_3d_read_ops((1,24,24,24,channel), (4,3,2,1,0), spatial_dims=(1,2,3), kernel_size=(2,2,2))
            layout_2 = ReadEstimatorV3.conv_3d_read_ops((1,24,24,24,channel), (3,2,1,4,0), spatial_dims=(1,2,3), kernel_size=(2,2,2))
            print(f"b = {batch}, c = {channel}, {layout_1} / {layout_2} = {layout_1 / layout_2}")
            ratios.append(layout_1 / layout_2)
    print(ratios)

    ratios = []
    fast_mode = False
    for b1, b2 in product([1, 2, 4, 8], [1, 2, 4, 8]):
        layout_1 = ReadEstimatorV3.dot_read_ops((b1, b2, 32, 64), (0, 1, 2, 3), reduce_dims=(2,3), fast=fast_mode)
        layout_2 = ReadEstimatorV3.dot_read_ops((b1, b2, 32, 64), (2, 3, 0, 1), reduce_dims=(2,3), fast=fast_mode)
        print(f"b = {[b1, b2]}, {layout_1} / {layout_2} = {layout_1 / layout_2}")
        ratios.append(layout_1 / layout_2)

if __name__ == '__main__':
    import time
    warnings.filterwarnings('ignore')

    single_file_eda()
    # create_dataset_feature(
    #     "/home/ron_zhu/TPU-Graph/datasets/TPUGraphsNpz/processed/xla_default*data*.pt",
    #     "/home/ron_zhu/data/tpugraphs/pb/pb/layout/xla/default/"
    # )

    # create_dataset_feature(
    #     "/home/ron/Projects/TPU-Graph/datasets/TPUGraphsNpz/processed/xla_default*data*.pt",
    #     "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/"
    # )
    # create_dataset_feature(
    #     "/home/ron/Projects/TPU-Graph/datasets/TPUGraphsNpz/processed/xla_random*data*.pt",
    #     "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/random/"
    # )
    # create_dataset_feature(
    #     "/home/ron/Projects/TPU-Graph/datasets/TPUGraphsNpz/processed/nlp_default*data*.pt",
    #     "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/nlp/default/"
    # )
    # create_dataset_feature(
    #     "/home/ron/Projects/TPU-Graph/datasets/TPUGraphsNpz/processed/nlp_random*data*.pt",
    #     "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/nlp/random/"
    # )

    # print(ReadEstimatorV1.conv_read_ops([2,24,24,24,64], [4,3,2,1,0], spatial_dims=[1,2,3], kernel_size=[2,2,2]))
    # print(ReadEstimatorV1.conv_read_ops([2,24,24,24,64], [3,2,1,4,0], spatial_dims=[1,2,3], kernel_size=[2,2,2]))
    # print(ReadEstimatorV1.conv_3d_read_ops([2,24,24,24,64], [4,3,2,1,0], spatial_dims=[1,2,3], kernel_size=[2,2,2]))
    # print(ReadEstimatorV1.conv_3d_read_ops([2,24,24,24,64], [3,2,1,4,0], spatial_dims=[1,2,3], kernel_size=[2,2,2]))
    
    # t0 = time.time()
    # print(ReadEstimatorV1.conv_2d_read_ops((2,24,24,64), (3,2,1,0), spatial_dims=(1,2), kernel_size=(2,2)))
    # print(time.time() - t0)
    # t0 = time.time()
    # print(ReadEstimatorV1.conv_2d_read_ops((2, 100, 100, 64), (3,0,2,1), spatial_dims=(1,2), kernel_size=(3,3), fast=True))
    # print(time.time() - t0)
    
    # print(ReadEstimatorV1.dot_read_ops([23, 64, 2048], [2, 1, 0], reduce_dims=[1,0]))
    # print(ReadEstimatorV1.dot_read_ops([23, 64, 2048], [1, 0, 2], reduce_dims=[1,0]))
    # print(ReadEstimatorV1.dot_read_ops([23, 64, 2048], [0, 1, 2], reduce_dims=[1,0]))
    # fast_mode_sanity_check()

    # print(ReadEstimatorV3.reshape_read_ops((4,2,3), (2,1,0), (8,3), (0, 1)))
    # print(ReadEstimatorV3.reshape_read_ops((40,20,30), (2,1,0), (2,20,600),(0,2,1)))