from yacs.config import CfgNode as CN
from torch_geometric.graphgym.register import register_config


@register_config('overwrite_defaults')
def overwrite_defaults_cfg(cfg):
    """Overwrite the default config values that are first set by GraphGym in
    torch_geometric.graphgym.config.set_cfg

    WARNING: At the time of writing, the order in which custom config-setting
    functions like this one are executed is random; see the referenced `set_cfg`
    Therefore never reset here config options that are custom added, only change
    those that exist in core GraphGym.
    """

    cfg.train.mode = 'custom'  # 'standard' uses PyTorch-Lightning since PyG 2.1
    cfg.device = 'cuda'  # 'standard' uses PyTorch-Lightning since PyG 2.1

    # Overwrite default dataset name
    cfg.dataset.name = 'none'

    # Overwrite default rounding precision
    cfg.round = 5
    cfg.train.auto_resume = True
    cfg.train.ckpt_clean = False


@register_config('extended_cfg')
def extended_cfg(cfg):
    """General extended config options.
    """

    # Additional name tag used in `run_dir` and `wandb_name` auto generation.
    cfg.name_tag = ""

    # In training, if True (and also cfg.train.enable_ckpt is True) then
    # always checkpoint the current best model based on validation performance,
    # instead, when False, follow cfg.train.eval_period checkpointing frequency.
    cfg.train.ckpt_best = True
    cfg.train.adap_margin = False
    
    cfg.train.regression = CN()
    cfg.train.regression.use = False
    cfg.train.regression.val_min = -1
    cfg.train.regression.val_max = -1
    cfg.train.regression.weight = 1.0

    cfg.train.gst = CN()
    cfg.train.gst.graph_embed_dims = 1
    cfg.train.gst.graph_embed_size = 1
    cfg.train.gst.sample_full_graph = False

    cfg.train.pair_rank = False

    cfg.dataset.source = 'nlp'
    cfg.dataset.search = 'random'
    cfg.dataset.tpu_task = 'layout'
    cfg.dataset.valid_for_train = []
    cfg.dataset.cache_in_memory = False
    cfg.dataset.inference_split = 'test'
    cfg.dataset.inference_num_config_cap = 1_000_000

    cfg.dataset.num_sample_config = 32
    cfg.dataset.eval_num_sample_config = 512
    cfg.dataset.config_sampler = ''

    cfg.dataset.input_feat_key = None
    cfg.dataset.extra_cfg_feat_keys = []
    cfg.dataset.extra_cfg_feat_dims = 0
    
    cfg.dataset.khop = CN()
    cfg.dataset.khop.use = False
    cfg.dataset.khop.hops = 1
    cfg.dataset.khop.bidirect = False

    cfg.debug = False
    cfg.model_ckpt = ''
    cfg.prefetch_factor = 2
