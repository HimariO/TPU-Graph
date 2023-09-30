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

    cfg.dataset.source = 'nlp'
    cfg.dataset.search = 'random'
    cfg.dataset.cache_in_memory = False
    cfg.dataset.num_sample_config = 32
    cfg.dataset.eval_num_sample_config = 512
    cfg.dataset.input_feat_key = None

    cfg.debug = False
    cfg.model_ckpt = ''
