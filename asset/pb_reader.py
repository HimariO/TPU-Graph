import os
from termcolor import colored

import hlo_proto_py.tuning_pb2 as tuning_pb
import hlo_proto_py.hlo_pb2 as hlo_pb

"""
ref on how different arugment of hlo insturction work:
https://www.tensorflow.org/xla/operation_semantics#reshape
"""

# pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/train/inference_mlperf_ssd_1200_batch_1.pb"
pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/valid/resnet_v1_50_official_batch_128_bf16.pb"

with open(pb_path, mode='rb') as f:
    hlo_obj = tuning_pb.ModuleTuningData()
    hlo_obj.ParseFromString(f.read())

    m = len(hlo_obj.module.computations)
    computes = hlo_obj.module.computations
    print(hlo_obj.module.name)
    print(m, computes[0])
    print('-' * 100)

    for i in range(m):
        comp: hlo_pb.HloComputationProto = computes[i]
        print(colored('*', color='green'), comp.name)
        for inst in comp.instructions:
            if 'reshape' in inst.name or '1convo' in inst.name:
                print(colored('>>', color='yellow'))
                print(inst)
    # breakpoint()
    # print(hlo_obj)