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