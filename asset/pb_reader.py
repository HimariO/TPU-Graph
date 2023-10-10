import os
from termcolor import colored
from collections import defaultdict
from typing import *

import hlo_proto_py.tuning_pb2 as tuning_pb
import hlo_proto_py.hlo_pb2 as hlo_pb

"""
ref on how different arugment of hlo insturction work:
https://www.tensorflow.org/xla/operation_semantics#reshape
"""

def single_file_eda():
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/train/inference_mlperf_ssd_1200_batch_1.pb"
    pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/valid/resnet_v1_50_official_batch_128_bf16.pb"
    # pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/nlp/default/valid/bert_multi_cased_L-12_H-768_A-12_batch_size_16_train.pb"

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
            # 'dot',
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
                    print("inst.outfeed_shape: ", inst.outfeed_shape)
                    # for k, p in enumerate(inst_chds[inst.id]):
                    for k, p in enumerate(inst.operand_ids):
                        print(colored(f"P[{k}]", color='cyan'), id2inst[p])
        # breakpoint()
        # print(hlo_obj)


def estimate_shape(inst: hlo_pb.HloInstructionProto, id2inst: Dict[int, hlo_pb.HloInstructionProto]) -> List[int]:
    feature = {
        'input_shape_1': [-1] * 6,
        'input_shape_2': [-1] * 6,
        'output_shape': [-1] * 6,
        'kernel_shape': [-1] * 6,
    }
    shape = list(inst.shape.dimensinos)
    default_layout = list(inst.shape.layout.minor_to_major)
    dtype = inst.shape.element_type
    if inst.opcode == "reshape":
        pass
    elif inst.opcode == "dot":
        pass
    elif inst.opcode == "convolution":
        hidden = id2inst[inst.operand_ids[0]]
        kernel = id2inst[inst.operand_ids[1]]

        for dim in inst.window.dimensions:
            dim.size
            dim.stride
            
            dim.padding_low
            dim.padding_high


if __name__ == '__main__':
    single_file_eda()