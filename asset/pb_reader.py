import hlo_proto_py.tuning_pb2 as tuning_pb

pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/train/inference_mlperf_ssd_1200_batch_1.pb"

with open(pb_path, mode='rb') as f:
    hlo_obj = tuning_pb.ModuleTuningData()
    hlo_obj.ParseFromString(f.read())

    print(hlo_obj.module.name)
    print(len(hlo_obj.module.computations), hlo_obj.module.computations[0])
    breakpoint()
    # print(hlo_obj)