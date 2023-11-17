from torch_geometric.graphgym.register import register_config


@register_config('custom_gnn')
def custom_gnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """

    # Use residual connections between the GNN layers.
    cfg.gnn.residual = False
    cfg.gnn.dim_feat_enc = 256
    cfg.gnn.enc_config = False
    cfg.gnn.enc_tile_config = False
    cfg.gnn.freeze_body = False
    cfg.gnn.dim_out = 1
    cfg.gnn.post_mp_norm = True
    cfg.gnn.avgmax_pooling = 'sum'
    cfg.gnn.cfg_feat_dim = 18
    cfg.gnn.cfg_feat_reweight = False
    cfg.gnn.late_fuse = False
    cfg.gnn.force_op_emb = False