```json
{
    'train_curve': {
        'epoch': [0, 1, 2, 3],
        'train_loss': [30.031482696533203, 29.277633666992188, 29.06572723388672, 28.81978416442871],
        'train_opa': [0.5523073077201843, 0.5845614671707153, 0.5722396969795227, 0.5907718539237976],
        'val_loss': [30.758089065551758, 29.826702117919922, 29.872278213500977, 30.472742080688477],
        'val_opa': [0.47976189851760864, 0.5226190686225891, 0.5571428537368774, 0.5642856955528259]
    },
    'final_opa': {},
    'args': {
        'source': 'xla',
        'search': 'random',
        'epochs': 10,
        'batch_size': 8,
        'configs': 16,
        'max_configs': 1000,
        'early_stop': 40,
        'keep_nodes': 5000,
        'learning_rate': 0.001,
        'clip_norm': 0.01,
        'out_dir': '~/out/tpugraphs_layout',
        'results_csv': '/home/ron_zhu/out/tpugraphs_layout/results_1694006089326_xla_random.csv',
        'validate_batches': 10,
        'run_id': ''
    }
} 

{
    'train_curve': {
        'epoch': [0, 1, 2, 3, 4, 5, 6, 7, 8],
        'train_loss': [30.61836814880371, 30.524171829223633, 30.432838439941406, 30.589670181274414, 30.460235595703125, 30.68816566467285, 30.273643493652344, 30.544910430908203, 30.49853515625],
        'train_opa': [0.5370218753814697, 0.5444110631942749, 0.5374419093132019, 0.5284230709075928, 0.5382618308067322, 0.5411314368247986, 0.535411536693573, 0.5290419459342957, 0.5032108426094055],
        'val_loss': [30.628009796142578, 30.88991355895996, 30.72537612915039, 30.758089065551758, 30.674203872680664, 30.554685592651367, 30.867904663085938, 30.56519889831543, 30.52389144897461],
        'val_opa': [0.5208581686019897, 0.4464285671710968, 0.4761904776096344, 0.48271751403808594, 0.5011904835700989, 0.5440475940704346, 0.4654761850833893, 0.5376344323158264, 0.5511904954910278]
    },
    'final_opa': {},
    'args': {
        'source': 'xla',
        'search': 'default',
        'epochs': 10,
        'batch_size': 8,
        'configs': 16,
        'max_configs': 1000,
        'early_stop': 40,
        'keep_nodes': 5000,
        'learning_rate': 0.001,
        'clip_norm': 0.01,
        'out_dir': '~/out/tpugraphs_layout',
        'results_csv': '/home/ron_zhu/out/tpugraphs_layout/results_1694168091930_xla_default.csv',
        'validate_batches': 10,
        'run_id': ''
    }
}

{
    'train_curve': {
        'epoch': [0, 1],
        'train_loss': [30.752939224243164, 30.713239669799805],
        'train_opa': [0.4928942322731018, 0.5056360960006714],
        'val_loss': [30.6994571685791, 30.656579971313477],
        'val_opa': [0.5095833539962769, 0.5400000214576721]
    },
    'final_opa': {},
    'args': {
        'source': 'nlp',
        'search': 'random',
        'epochs': 10,
        'batch_size': 8,
        'configs': 16,
        'max_configs': 1000,
        'early_stop': 40,
        'keep_nodes': 5000,
        'learning_rate': 0.001,
        'clip_norm': 0.01,
        'out_dir': '~/out/tpugraphs_layout',
        'results_csv': '/home/ron_zhu/out/tpugraphs_layout/results_1694491098109_nlp_random.csv',
        'validate_batches': 10,
        'run_id': ''
    }
}

{
    'train_curve': {
        'epoch': [0, 1, 2],
        'train_loss': [30.688444137573242, 30.710844039916992, 30.699859619140625],
        'train_opa': [0.5161670446395874, 0.500694751739502, 0.5128636956214905],
        'val_loss': [30.720687866210938, 30.70270347595215, 30.67327880859375],
        'val_opa': [0.4983333349227905, 0.5085452198982239, 0.5181325674057007]
    },
    'final_opa': {},
    'args': {
        'source': 'nlp',
        'search': 'default',
        'epochs': 10,
        'batch_size': 8,
        'configs': 16,
        'max_configs': 1000,
        'early_stop': 40,
        'keep_nodes': 5000,
        'learning_rate': 0.001,
        'clip_norm': 0.01,
        'out_dir': '~/out/tpugraphs_layout',
        'results_csv': '/home/ron_zhu/out/tpugraphs_layout/results_1694501300765_nlp_default.csv',
        'validate_batches': 10,
        'run_id': ''
    }
}
```

```
UPDATE-BASELINE:
xla-random:
    best-val-opa: 0.5924
xla-default:
    best-val-opa: 0.5607
nlp-random:
    best-val-opa: 0.86
nlp-default:
    best-val-opa: 0.66

xla-tail:
    val-opa: 0.8673

.17 + .13 + .17 + .11 + .12
```

```shell

xla-def enselble test 11/1:
    * opa 73/seed 0: tests/xla-default-sage-fullenc-khop-extra/tpu-khop-extra/test_20231018_1697559303.pt
    * opa 73/seed 5: tests/xla-default-sage-fullenc-khop-extra/tpu-khop-extra/test_20231101_1698825964.pt
'["tests/xla-default-sage-fullenc-khop-extra/tpu-khop-extra/test_20231018_1697559303.pt","tests/xla-default-sage-fullenc-khop-extra/tpu-khop-extra/test_20231101_1698825964.pt"]'
```

### Tricks Performance Compartion

|           | GST+EX2 | Full Graph | MixSearch  |   |
|-----------|---------|------------|------------|---|
| XLA-DEF   | 73*     | "          | 79         |   |
| XLA-RAND  | 75      | 78         | 92         |   |
| NLP-DEF   | 77      | "          | "          |   |
| NLP-RAND  | 94      | 97*        | "          |   |

```py
# xla-rand only
{'bert_pretraining.4x4.fp16': 0.8631889763779528,
 'inception_v3_batch_128_train': 0.6839320866141733,
 'mlperf_bert_batch_24_2x2': 0.8222194881889764,
 'resnet50.4x4.fp16': 0.8955462598425197,
 'resnet_v1_50_official_batch_128_bf16': 0.6740895669291339,
 'tf2_bert_pretrain_dynamic_batch_size': 0.905880905511811,
 'unet_3d.4x4.bf16': 0.5915354330708661}

# xla-rand mixing with nlp-rand
{'bert_pretraining.4x4.fp16': 0.9277805118110236,
 'inception_v3_batch_128_train': 0.921136811023622,
 'mlperf_bert_batch_24_2x2': 0.9056348425196851,
 'resnet50.4x4.fp16': 0.9261811023622047,
 'resnet_v1_50_official_batch_128_bf16': 0.6941437007874016,
 'tf2_bert_pretrain_dynamic_batch_size': 0.9251968503937008,
 'unet_3d.4x4.bf16': 0.8567913385826772}


 # >>> checkpoint with fixed MixTPU dataset
 
 # config using MixTPU on local
{
    'resnet_v1_50_official_batch_128_bf16': 0.9076033464566929,
    'bert_pretraining.4x4.fp16': 0.9608759842519685,
    'tf2_bert_pretrain_dynamic_batch_size': 0.952263779527559,
    'unet_3d.4x4.bf16': 0.88939468503937,
    'resnet50.4x4.fp16': 0.9650590551181102,
    'inception_v3_batch_128_train': 0.9472194881889764,
    'mlperf_bert_batch_24_2x2': 0.9349163385826772
}

 # config using TPUNpz on local
 {
    'resnet_v1_50_official_batch_128_bf16': 0.9076033464566929,
    'bert_pretraining.4x4.fp16': 0.9608759842519685,
    'tf2_bert_pretrain_dynamic_batch_size': 0.952386811023622,
    'unet_3d.4x4.bf16': 0.8916092519685039,
    'resnet50.4x4.fp16': 0.9650590551181102,
    'inception_v3_batch_128_train': 0.9472194881889764,
    'mlperf_bert_batch_24_2x2': 0.9349163385826772
}

# XLA-Default
# without mixing xla-random, opa: 0.8001
{
    'resnet_v1_50_official_batch_128_bf16': 0.8753292361720808,
    'bert_pretraining.4x4.fp16': 0.6613292388681729,
    'tf2_bert_pretrain_dynamic_batch_size': 0.8453947368421053,
    'resnet50.4x4.fp16': 0.8148310662571303,
    'inception_v3_batch_128_train': 0.8574561403508771,
    'mlperf_bert_batch_24_2x2': 0.8622886009224687,
    'unet_3d.4x4.bf16': 0.6875549692172384,
}

# with xla-random concat to graph config, opa:0.7832
{
    'resnet_v1_50_official_batch_128_bf16': 0.7578947368421053,
    'bert_pretraining.4x4.fp16': 0.6326754385964912,
    'tf2_bert_pretrain_dynamic_batch_size': 0.8326754385964912,
    'resnet50.4x4.fp16': 0.8072368421052631,
    'inception_v3_batch_128_train': 0.8923245614035088,
    'mlperf_bert_batch_24_2x2': 0.8392543859649123,
    'unet_3d.4x4.bf16': 0.7203947368421053,
}

```


```
XLA-subset model types:
* CNN
alexnet_train_batch_32.npz
efficientnet_b7_eval_batch_1.npz
inception_v2_batch_128_train.npz
inception_v2_batch_8_train.npz
inception_v3_batch_8_train.npz
inference_mlperf_resnet_batch_16.npz
inference_mlperf_resnet_batch_256.npz
inference_mlperf_ssd_1200_batch_128.npz
inference_mlperf_ssd_1200_batch_1.npz
inference_mlperf_ssd_1200_batch_2.npz
mask_rcnn_batch_16_bf16_img1024.npz
mask_rcnn_batch_4_bf16_img1408.npz
mask_rcnn_resnet50.4x4.bf16.performance.npz
mlperf_maskrcnn_1_shard_batch_4.npz
mlperf_maskrcnn_batch_2.npz
mlperf_maskrcnn_batch_4.npz
mlperf_resnet.npz
mlperf_ssd_1_shard_batch_8_fast_epoch.npz
mlperf_ssd_2_shard_batch_8_fast_epoch.npz
mnasnet_a1_batch_128.npz
mnasnet_b1_batch_128.npz
resnet50.2x2.fp16.npz
resnet50.2x2.fp32.npz
resnet50_3d.2x2.bf16.npz
resnet50.4x4.bf16.npz
resnet50.4x4.bf16.performance.npz
resnet50.8x16.fp16.npz
resnet50.8x8.fp16.npz
resnet50.8x8.fp32.npz
resnet_v1_50_official_batch_128_f32.npz
resnet_v1_50_official_batch_32_bf16.npz
resnet_v2_101_batch_128.npz
resnet_v2_152_batch_128.npz
resnet_v2_152_batch_64.npz
resnet_v2_200_batch_32.npz
resnet_v2_200_batch_64.npz
resnet_v2_50_batch_128.npz
resnet_v2_50_batch_16.npz
retinanet.2x2.fp32.npz
retinanet.4x4.bf16.performance.npz
retinanet.4x4.fp32.npz
shapemask.4x4.fp32.npz  // segmentation
unet3d.npz
xception_imagenet.npz

* attention model
bert_classifier.2x2.fp32.npz
bert_classifier.2x2.fp32.performance.npz
bert_pretraining.2x2.fp16.npz
bert_pretraining.8x16.fp16.npz
bert_pretraining.8x8.fp32.performance.npz
bert_squad.2x2.fp32.npz
mlperf_transformer.npz
ncf.2x2.fp32.npz   // Neural Collaborative Filtering
tf2_bert_pretrain_dynamic_sequence_length.npz
tf2_bert_squad_dynamic.npz
transformer.2x2.fp32.npz
transformer.4x4.bf16.npz
transformer.4x4.fp16.npz
transformer.4x4.fp32.performance.npz
transformer_tf2_dynamic_shape.npz
trax_lsh_attention.npz

* RNN
brax_es.npz
magenta_dynamic.npz
magenta.npz
mlperf_nmt_1_shard_batch_8.npz
mlperf_nmt_batch_64.npz
mlperf_nmt_batch_8.npz
mlperf_resnet_batch_128_1_shard.npz
openai_v0_rnn_natural.npz
openai_v0_rnn_optimized.npz
```

```py
pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/test/05ae41e26dd3c4c06390371a0423233c.pb"  # eff-b7
pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/test/5335ed13823b0a518ee3c79ba4425f34.pb"  # eff-b7
pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/test/937ee0eb0d5d6151b7b8252933b5c1c9.pb"  # resnet 50
pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/test/db59a991b7c607634f13570d52ce885f.pb"  # conv net
pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/test/fbaa8bb6a1aed9988281085c91065c05.pb"  # self_suppresion -> nms/(https://github.com/tensorflow/tpu/blob/master/models/official/detection/ops/nms.py)
pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/test/cd708819d3f5103afd6460b15e74eaf3.pb"  # MLP
pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/test/3e7156ac468dfb75cf5c9615e1e5887d.pb"  # bert
pb_path = "/home/ron/Projects/TPU-Graph/datasets/pb/pb/layout/xla/default/test/e8a3a1401b5e79f66d7037e424f3b6df.pb"  # bert_classifier/sentence_prediction
```


## NLP test to training set graph mapping

```
>> /root/data/tpugraphs/npz/layout/nlp/default/test/b2fdde3b72980907578648774101543e.npz
(1000, 72, 18)
(72,)
[('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-6_H-768_A-12_batch_size_32_test.npz',
  (72, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-6_H-768_A-12_batch_size_64_test.npz',
  (72, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-6_H-256_A-4_batch_size_16_test.npz',
  (72, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-6_H-128_A-2_batch_size_64_test.npz',
  (72, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-6_H-512_A-8_batch_size_32_test.npz',
  (72, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-6_H-512_A-8_batch_size_16_test.npz',
  (72, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-6_H-128_A-2_batch_size_32_test.npz',
  (72, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-6_H-256_A-4_batch_size_64_test.npz',
  (72, 18))]

>> /root/data/tpugraphs/npz/layout/nlp/default/test/29886a50d55cfe77a9497bc906c76ce9.npz
(1000, 152, 18)
(152,)
[('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-4_H-128_A-2_batch_size_32_train.npz',
  (152, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-4_H-128_A-2_batch_size_16_train.npz',
  (152, 18))]

>> /root/data/tpugraphs/npz/layout/nlp/default/test/7105451001e119f65b66570d170b94a8.npz
(1000, 104, 18)
(104,)
[('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-2_H-768_A-12_batch_size_32_train.npz',
  (104, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-2_H-256_A-4_batch_size_64_train.npz',
  (104, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-2_H-768_A-12_batch_size_16_train.npz',
  (104, 18))]

>> /root/data/tpugraphs/npz/layout/nlp/default/test/171b0513d8874a427ccfa46d136fbadc.npz
(1000, 344, 18)
(344,)
[('/root/data/tpugraphs/npz/layout/nlp/random/train/electra_base_batch_size_16_train.npz',
  (344, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/experts_wiki_books_batch_size_32_train.npz',
  (344, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/experts_wiki_books_batch_size_64_train.npz',
  (344, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/bert_en_uncased_L-12_H-768_A-12_batch_size_64_train.npz',
  (344, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/bert_en_uncased_L-12_H-768_A-12_batch_size_32_train.npz',
  (344, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/bert_en_cased_L-12_H-768_A-12_batch_size_16_train.npz',
  (344, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/electra_base_batch_size_64_train.npz',
  (344, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/electra_base_batch_size_32_train.npz',
  (344, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/bert_en_cased_L-12_H-768_A-12_batch_size_64_train.npz',
  (344, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/experts_pubmed_batch_size_32_train.npz',
  (344, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/bert_multi_cased_L-12_H-768_A-12_batch_size_32_train.npz',
  (344, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-256_A-4_batch_size_64_train.npz',
  (344, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/experts_wiki_books_batch_size_16_train.npz',
  (344, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-768_A-12_batch_size_32_train.npz',
  (344, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/bert_multi_cased_L-12_H-768_A-12_batch_size_64_train.npz',
  (344, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-256_A-4_batch_size_32_train.npz',
  (344, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-768_A-12_batch_size_16_train.npz',
  (344, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/bert_en_cased_L-12_H-768_A-12_batch_size_32_train.npz',
  (344, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/bert_en_uncased_L-12_H-768_A-12_batch_size_16_train.npz',
  (344, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/experts_pubmed_batch_size_64_train.npz',
  (344, 18))]

>> /root/data/tpugraphs/npz/layout/nlp/default/test/60880ed76de53f4d7a1b960b24f20f7d.npz
(1000, 120, 18)
(120,)
[('/root/data/tpugraphs/npz/layout/nlp/random/train/bert_en_uncased_L-12_H-768_A-12_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-768_A-12_batch_size_64_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-128_A-2_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-512_A-8_batch_size_64_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/experts_pubmed_batch_size_64_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-256_A-4_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/bert_en_uncased_L-12_H-768_A-12_batch_size_64_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/experts_wiki_books_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-128_A-2_batch_size_64_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/experts_wiki_books_batch_size_16_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/bert_en_cased_L-12_H-768_A-12_batch_size_64_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/bert_en_uncased_L-12_H-768_A-12_batch_size_16_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/bert_multi_cased_L-12_H-768_A-12_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/experts_wiki_books_batch_size_64_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-256_A-4_batch_size_64_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-256_A-4_batch_size_16_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-768_A-12_batch_size_16_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/electra_base_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/electra_base_batch_size_64_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-512_A-8_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/bert_en_cased_L-12_H-768_A-12_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-128_A-2_batch_size_16_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-768_A-12_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/experts_pubmed_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-512_A-8_batch_size_16_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/bert_multi_cased_L-12_H-768_A-12_batch_size_64_test.npz',
  (120, 18))]

>> /root/data/tpugraphs/npz/layout/nlp/default/test/58cc2e418c3a8a19b871e15964b534ad.npz
(1000, 88, 18)
(88,)
[('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-512_A-8_batch_size_16_test.npz',
  (88, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-256_A-4_batch_size_32_test.npz',
  (88, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-128_A-2_batch_size_16_test.npz',
  (88, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-768_A-12_batch_size_32_test.npz',
  (88, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-768_A-12_batch_size_64_test.npz',
  (88, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-128_A-2_batch_size_32_test.npz',
  (88, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-768_A-12_batch_size_16_test.npz',
  (88, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-512_A-8_batch_size_32_test.npz',
  (88, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-256_A-4_batch_size_64_test.npz',
  (88, 18))]

>> /root/data/tpugraphs/npz/layout/nlp/default/test/f6c146fc5cf10be4f3accbaca9897311.npz
(1000, 200, 18)
(200,)
[('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-6_H-128_A-2_batch_size_32_train.npz',
  (200, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-6_H-128_A-2_batch_size_64_train.npz',
  (200, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-6_H-128_A-2_batch_size_16_train.npz',
  (200, 18))]

>> /root/data/tpugraphs/npz/layout/nlp/default/test/38524e2ff135ded55b5286407e7af6b7.npz
(1000, 120, 18)
(120,)
[('/root/data/tpugraphs/npz/layout/nlp/random/train/bert_en_uncased_L-12_H-768_A-12_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-768_A-12_batch_size_64_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-128_A-2_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-512_A-8_batch_size_64_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/experts_pubmed_batch_size_64_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-256_A-4_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/bert_en_uncased_L-12_H-768_A-12_batch_size_64_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/experts_wiki_books_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-128_A-2_batch_size_64_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/experts_wiki_books_batch_size_16_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/bert_en_cased_L-12_H-768_A-12_batch_size_64_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/bert_en_uncased_L-12_H-768_A-12_batch_size_16_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/bert_multi_cased_L-12_H-768_A-12_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/experts_wiki_books_batch_size_64_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-256_A-4_batch_size_64_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-256_A-4_batch_size_16_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-768_A-12_batch_size_16_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/electra_base_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/electra_base_batch_size_64_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-512_A-8_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/bert_en_cased_L-12_H-768_A-12_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-128_A-2_batch_size_16_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-768_A-12_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/experts_pubmed_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-512_A-8_batch_size_16_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/bert_multi_cased_L-12_H-768_A-12_batch_size_64_test.npz',
  (120, 18))]

>> /root/data/tpugraphs/npz/layout/nlp/default/test/3a0c5517a87df8d82fd637b83298a3ba.npz
(1001, 848, 18)
(848,)
[]

>> /root/data/tpugraphs/npz/layout/nlp/default/test/6c1101f6231f4d1722c3b9f6d1e25026.npz
(1000, 104, 18)
(104,)
[('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-2_H-128_A-2_batch_size_16_train.npz',
  (104, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-2_H-512_A-8_batch_size_32_train.npz',
  (104, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-2_H-128_A-2_batch_size_64_train.npz',
  (104, 18))]

>> /root/data/tpugraphs/npz/layout/nlp/default/test/016ac66a44a906a695afd2228509046a.npz
(1000, 88, 18)
(88,)
[('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-512_A-8_batch_size_16_test.npz',
  (88, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-256_A-4_batch_size_32_test.npz',
  (88, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-128_A-2_batch_size_16_test.npz',
  (88, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-768_A-12_batch_size_32_test.npz',
  (88, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-768_A-12_batch_size_64_test.npz',
  (88, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-128_A-2_batch_size_32_test.npz',
  (88, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-768_A-12_batch_size_16_test.npz',
  (88, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-512_A-8_batch_size_32_test.npz',
  (88, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-256_A-4_batch_size_64_test.npz',
  (88, 18))]

>> /root/data/tpugraphs/npz/layout/nlp/default/test/492c7a94d559aa4a88769142d2a68362.npz
(1000, 248, 18)
(248,)
[('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-768_A-12_batch_size_16_train.npz',
  (248, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-256_A-4_batch_size_32_train.npz',
  (248, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-256_A-4_batch_size_64_train.npz',
  (248, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-768_A-12_batch_size_64_train.npz',
  (248, 18))]

>> /root/data/tpugraphs/npz/layout/nlp/default/test/d15316c12eefdef1ba549eb433797f77.npz
(1000, 88, 18)
(88,)
[('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-512_A-8_batch_size_16_test.npz',
  (88, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-256_A-4_batch_size_32_test.npz',
  (88, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-128_A-2_batch_size_16_test.npz',
  (88, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-768_A-12_batch_size_32_test.npz',
  (88, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-768_A-12_batch_size_64_test.npz',
  (88, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-128_A-2_batch_size_32_test.npz',
  (88, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-768_A-12_batch_size_16_test.npz',
  (88, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-512_A-8_batch_size_32_test.npz',
  (88, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-8_H-256_A-4_batch_size_64_test.npz',
  (88, 18))]

>> /root/data/tpugraphs/npz/layout/nlp/default/test/7f6284ebe027b1e9a3850fc703858a59.npz
(1000, 120, 18)
(120,)
[('/root/data/tpugraphs/npz/layout/nlp/random/train/bert_en_uncased_L-12_H-768_A-12_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-768_A-12_batch_size_64_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-128_A-2_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-512_A-8_batch_size_64_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/experts_pubmed_batch_size_64_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-256_A-4_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/bert_en_uncased_L-12_H-768_A-12_batch_size_64_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/experts_wiki_books_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-128_A-2_batch_size_64_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/experts_wiki_books_batch_size_16_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/bert_en_cased_L-12_H-768_A-12_batch_size_64_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/bert_en_uncased_L-12_H-768_A-12_batch_size_16_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/bert_multi_cased_L-12_H-768_A-12_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/experts_wiki_books_batch_size_64_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-256_A-4_batch_size_64_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-256_A-4_batch_size_16_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-768_A-12_batch_size_16_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/electra_base_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/electra_base_batch_size_64_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-512_A-8_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/bert_en_cased_L-12_H-768_A-12_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-128_A-2_batch_size_16_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-768_A-12_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/experts_pubmed_batch_size_32_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-12_H-512_A-8_batch_size_16_test.npz',
  (120, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/bert_multi_cased_L-12_H-768_A-12_batch_size_64_test.npz',
  (120, 18))]

>> /root/data/tpugraphs/npz/layout/nlp/default/test/32531d07a084b319dce484f53a4cf3fc.npz
(1000, 152, 18)
(152,)
[('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-4_H-512_A-8_batch_size_64_train.npz',
  (152, 18))]

>> /root/data/tpugraphs/npz/layout/nlp/default/test/23559853d9702baaaacbb0c83fd32266.npz
(1000, 56, 18)
(56,)
[('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-4_H-768_A-12_batch_size_32_test.npz',
  (56, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-4_H-256_A-4_batch_size_16_test.npz',
  (56, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-4_H-128_A-2_batch_size_64_test.npz',
  (56, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-4_H-512_A-8_batch_size_64_test.npz',
  (56, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-4_H-128_A-2_batch_size_32_test.npz',
  (56, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-4_H-512_A-8_batch_size_16_test.npz',
  (56, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-4_H-256_A-4_batch_size_64_test.npz',
  (56, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-4_H-512_A-8_batch_size_32_test.npz',
  (56, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-4_H-256_A-4_batch_size_32_test.npz',
  (56, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/small_bert_bert_en_uncased_L-4_H-768_A-12_batch_size_64_test.npz',
  (56, 18))]

>> /root/data/tpugraphs/npz/layout/nlp/default/test/71b79ca6db513e7979c3702c595150c2.npz
(1000, 288, 18)
(288,)
[('/root/data/tpugraphs/npz/layout/nlp/random/train/talking-heads_large_batch_size_32_test.npz',
  (288, 18)),
 ('/root/data/tpugraphs/npz/layout/nlp/random/train/talking-heads_large_batch_size_16_test.npz',
  (288, 18))]
```



```py
# concat nlp-rand+def (512 dim, 3 post_mp), no interval sample
{'albert_en_xlarge_batch_size_16_test': 0.8127192982456141,
 'bert_en_cased_L-12_H-768_A-12_batch_size_16_test': 0.8695175438596491,
 'bert_multi_cased_L-12_H-768_A-12_batch_size_16_train': 0.6535087719298246,
 'small_bert_bert_en_uncased_L-10_H-128_A-2_batch_size_32_test': 0.836954136493307,
 'small_bert_bert_en_uncased_L-10_H-128_A-2_batch_size_64_train': 0.8508771929824561,
 'small_bert_bert_en_uncased_L-10_H-256_A-4_batch_size_32_test': 0.8776315789473684,
 'small_bert_bert_en_uncased_L-10_H-256_A-4_batch_size_64_train': 0.7986842105263158,
 'small_bert_bert_en_uncased_L-10_H-512_A-8_batch_size_64_test': 0.7526315789473684,
 'small_bert_bert_en_uncased_L-10_H-768_A-12_batch_size_16_train': 0.8115814871682386,
 'small_bert_bert_en_uncased_L-10_H-768_A-12_batch_size_32_test': 0.8335526315789473,
 'small_bert_bert_en_uncased_L-12_H-768_A-12_batch_size_64_train': 0.6089054617240623,
 'small_bert_bert_en_uncased_L-2_H-256_A-4_batch_size_32_train': 0.8124588725597719,
 'small_bert_bert_en_uncased_L-4_H-256_A-4_batch_size_32_train': 0.7980263157894737,
 'small_bert_bert_en_uncased_L-4_H-512_A-8_batch_size_32_train': 0.6094298245614035,
 'small_bert_bert_en_uncased_L-6_H-256_A-4_batch_size_16_train': 0.8072368421052631,
 'small_bert_bert_en_uncased_L-6_H-256_A-4_batch_size_64_train': 0.6964912280701754,
 'small_bert_bert_en_uncased_L-6_H-512_A-8_batch_size_64_test': 0.8019736842105263,
 'small_bert_bert_en_uncased_L-6_H-768_A-12_batch_size_16_test': 0.6787280701754386,
 'small_bert_bert_en_uncased_L-6_H-768_A-12_batch_size_32_train': 0.7350877192982456,
 'talking-heads_large_batch_size_16_train': 0.8513157894736842}


# nlp-def only (bag of tricks, 256 dim, 0.1 dropout), no interval sample
{'albert_en_xlarge_batch_size_16_test': 0.6765350877192983, 
 'bert_en_cased_L-12_H-768_A-12_batch_size_16_test': 0.8703947368421052,
 'bert_multi_cased_L-12_H-768_A-12_batch_size_16_train': 0.6828947368421052, #
 'small_bert_bert_en_uncased_L-10_H-128_A-2_batch_size_32_test': 0.8204959403116086,
 'small_bert_bert_en_uncased_L-10_H-128_A-2_batch_size_64_train': 0.7918859649122807, 
 'small_bert_bert_en_uncased_L-10_H-256_A-4_batch_size_32_test': 0.8401315789473685, 
 'small_bert_bert_en_uncased_L-10_H-256_A-4_batch_size_64_train': 0.8192982456140351, 
 'small_bert_bert_en_uncased_L-10_H-512_A-8_batch_size_64_test': 0.7407894736842106, 
 'small_bert_bert_en_uncased_L-10_H-768_A-12_batch_size_16_train': 0.8041237113402062, 
 'small_bert_bert_en_uncased_L-10_H-768_A-12_batch_size_32_test': 0.8230263157894737,
 'small_bert_bert_en_uncased_L-12_H-768_A-12_batch_size_64_train': 0.7622285588944944, #
 'small_bert_bert_en_uncased_L-2_H-256_A-4_batch_size_32_train': 0.7620092125466111,
 'small_bert_bert_en_uncased_L-4_H-256_A-4_batch_size_32_train': 0.819078947368421, #
 'small_bert_bert_en_uncased_L-4_H-512_A-8_batch_size_32_train': 0.8043859649122806, #
 'small_bert_bert_en_uncased_L-6_H-256_A-4_batch_size_16_train': 0.8304824561403509, #
 'small_bert_bert_en_uncased_L-6_H-256_A-4_batch_size_64_train': 0.6883771929824561,
 'small_bert_bert_en_uncased_L-6_H-512_A-8_batch_size_64_test': 0.8199561403508772, 
 'small_bert_bert_en_uncased_L-6_H-768_A-12_batch_size_16_test': 0.6699561403508771,
 'small_bert_bert_en_uncased_L-6_H-768_A-12_batch_size_32_train': 0.7552631578947369, 
 'talking-heads_large_batch_size_16_train': 0.8116228070175439}
```