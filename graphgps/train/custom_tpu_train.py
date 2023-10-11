import os
import logging
import time
import copy
import datetime
from typing import *
from collections import defaultdict
from contextlib import nullcontext

import pysnooper
import numpy as np
import torch
import torch_geometric.nn as tnn
import pandas as pd
from torch import nn, Tensor
from torch_geometric.data import Batch
from torch_geometric.graphgym.checkpoint import load_ckpt, save_ckpt, clean_ckpt
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.loss import compute_loss
from torch_geometric.graphgym.register import register_train
from torch_geometric.graphgym.utils.epoch import is_eval_epoch, is_ckpt_epoch
from torch_geometric.data import Data
from tqdm import tqdm
from loguru import logger as ulogger

from graphgps.loss.subtoken_prediction_loss import subtoken_cross_entropy
from graphgps.utils import cfg_to_dict, flatten_dict, make_wandb_name
from graphgps.history import History
from graphgps.train.gst_utils import (
    batch_sample_graph_segs,
    cached_node_embed,
    TPUModel,
    CheckpointWrapper,
)


def pairwise_hinge_loss_batch(pred, true):
    # pred: (batch_size, num_preds )
    # true: (batch_size, num_preds)
    batch_size = pred.shape[0]
    num_preds = pred.shape[1]
    i_idx = torch.arange(num_preds).repeat(num_preds)
    j_idx = torch.arange(num_preds).repeat_interleave(num_preds)
    pairwise_true = true[:,i_idx] > true[:,j_idx]
    loss = torch.sum(torch.nn.functional.relu(0.1 - (pred[:,i_idx] - pred[:,j_idx])) * pairwise_true.float()) / batch_size
    return loss


def pairwise_hinge_loss(pred, true):
    num_preds = pred.shape[0]
    i_idx = torch.arange(num_preds).repeat(num_preds)
    j_idx = torch.arange(num_preds).repeat_interleave(num_preds)
    pairwise_true = true[i_idx] > true[j_idx]
    loss = torch.sum(torch.nn.functional.relu(0.1 - (pred[i_idx] - pred[j_idx])) * pairwise_true.float())
    opa_indices = pairwise_true.nonzero().flatten()
    opa_preds = pred[i_idx[opa_indices]] - pred[j_idx[opa_indices]]
    return loss, pairwise_true[opa_indices], opa_preds


@ulogger.catch(reraise=True)
def train_epoch(logger, loader, model: TPUModel, optimizer, scheduler, emb_table: History, batch_accumulation: int):
    model.train()
    optimizer.zero_grad()
    time_start = time.time()
    num_sample_config = cfg.dataset.num_sample_config  # number of configs per graph

    if cfg.debug: print(f"@ Start of Epoch")

    for iter, batch in enumerate(loader):
        if cfg.debug: print(f"@ iter-{iter} start")
        batch, sampled_idx = batch
        
        t0 = time.time()
        # with pysnooper.snoop():
        if isinstance(batch, Batch):
            batch.to(torch.device(cfg.device))
            true = batch.y
            batch_list = batch.to_data_list()
            (
                batch_train_list,
                batch_other,
                batch_num_parts,
                segments_to_train,
            ) = batch_sample_graph_segs(
                batch_list, sampled_idx, emb_table, 
                num_sample_config=num_sample_config
            )
        else:
            (   
                batch_obj,
                batch_list,
                batch_train_list,
                batch_num_parts,
                segments_to_train,
            ) = batch
            
            batch_obj.to(torch.device(cfg.device))
            true = batch_obj.y
            batch_other = cached_node_embed(batch_list, sampled_idx, segments_to_train, emb_table)
        
        if cfg.debug: print(f"@ iter-{iter} graph: ", set([b.graph_name for b in batch_train_list]))
        td0 = time.time() - t0
        t1 = time.time()

        batch_train = Batch.from_data_list(batch_train_list)
        batch_train = batch_train.to(torch.device(cfg.device))
        true = true.to(torch.device(cfg.device))  # (batch_size * num_sample,)
        """
        concat node features & linear project to lower dim
        """
        batch_train = model.gather_input_feat(batch_train)
        
        """
        Inference on sampled graph segments
        """
        custom_gnn = model.model.model  # TPUModel.GraphGymModule.GNN
        module_len = len(list(custom_gnn.children()))
        predict_ctx = torch.no_grad() if cfg.gnn.freeze_body else nullcontext()

        with predict_ctx:
            for i, module in enumerate(custom_gnn.children()):
                if i < module_len - 1:
                    batch_train = module(batch_train)
                if i == module_len - 1:
                    batch_train_embed = tnn.global_max_pool(batch_train.x, batch_train.batch) \
                                    + tnn.global_mean_pool(batch_train.x, batch_train.batch)
        
        graph_embed = batch_train_embed / torch.norm(batch_train_embed, dim=-1, keepdim=True)
        for i, module in enumerate(custom_gnn.children()):
            if i == module_len - 1:
                graph_embed = module.layer_post_mp(graph_embed)
        
        td1 = time.time() - t1
        t2 = time.time()

        binomial = torch.distributions.binomial.Binomial(probs=0.5)
        if batch_other:
            """
            Sample some cached embedding of graph segements, 
            use mean of cached + inferenced embedding as full-graph embedding.
            """
            batch_other = torch.cat(batch_other, dim=0)
            mask =  binomial.sample((batch_other.shape[0], 1)).to(torch.device(cfg.device))
            batch_other = batch_other.to(torch.device(cfg.device))
            batch_other = batch_other * mask    
            batch_other_embed = torch.zeros_like(graph_embed)
            part_cnt = 0
            for i, num_parts in enumerate(batch_num_parts):
                m = num_parts - 1
                batch_other_embed[i, :] += torch.sum(
                    batch_other[part_cnt: part_cnt + m, :],
                    dim=0, 
                    keepdim=False
                )
                part_cnt += m
            batch_num_parts = torch.Tensor(batch_num_parts).to(torch.device(cfg.device))
            batch_num_parts = batch_num_parts.view(-1, 1)
            multiplier_num = (batch_num_parts - 1)/ 2 + 1
            pred = graph_embed * multiplier_num + batch_other_embed
        else:
            pred = graph_embed
        
        """
        Compute loss
        """
        if 'TPUGraphs' in cfg.dataset.name:
            pred = pred.view(-1, num_sample_config)
            true = true.view(-1, num_sample_config)
            loss = pairwise_hinge_loss_batch(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred.detach().to('cpu', non_blocking=True)
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to('cpu', non_blocking=True)
            _pred = pred_score.detach().to('cpu', non_blocking=True)
        loss.backward()

        td2 = time.time() - t2
        td = time.time() - t0
        toms = lambda f: f"{f * 1000:.2f} ms"
        if cfg.debug:
            print(toms(td0), toms(td1), toms(td2), toms(td))
        
        # Parameters update after accumulating gradients for given num. batches.
        if ((iter + 1) % batch_accumulation == 0) or (iter + 1 == len(loader)):
            if cfg.optim.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        for i in range(graph_embed.shape[0]):
            b = i // num_sample_config  # batch index
            src_graph = batch_list[b]
            flat_ind = (
                src_graph.partition_idx.cpu() + 
                sampled_idx[b][i % num_sample_config] * (len(src_graph.partptr) - 1) + 
                segments_to_train[b]
            )
            emb_table.push(graph_embed[i].cpu(), flat_ind)
        
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name)
        time_start = time.time()
        if cfg.debug: print(f"@ iter{iter} end")


@torch.no_grad()
def eval_epoch(logger, loader, model: TPUModel, split='val'):
    model.eval()
    time_start = time.time()
    num_sample_config = cfg.dataset.eval_num_sample_config
    
    loader_bar = tqdm(loader)
    loader_bar.set_description_str('Eval Epoch')
    rankings = {} # defaultdict(list)
    
    for batch in loader_bar:
        # batch, _ = preprocess_batch(batch, model, num_sample_config)
        batch, sampled_idx = batch
        batch: Batch
        sampled_idx: List[Tensor]
        """
        NOTE: 
        to be able to support inference on all node configs per graph ontest-set,
        `num_sample_config` will be dynamicly capped by `graph.number_config` per batch.
        """
        batch_num_sample = len(sampled_idx[0])
        
        batch.split = split
        true = batch.y
        batch_list = batch.to_data_list()
        batch_seg = []  # each `Data` in the list = one segment from one graph's one config`
        batch_num_parts = []

        for i in range(len(batch_list)):
            partptr = batch_list[i].partptr.cpu().numpy()
            num_parts = len(partptr) - 1
            batch_num_parts.append(num_parts)
            for j in range(num_parts):
                start = int(partptr[j])
                length = int(partptr[j + 1]) - start

                N, E = batch_list[i].num_nodes, batch_list[i].num_edges
                data = copy.copy(batch_list[i])
                del data.num_nodes
                adj, data.adj = data.adj, None

                adj = adj.narrow(0, start, length).narrow(1, start, length)
                edge_idx = adj.storage.value()

                for key, item in data:
                    if isinstance(item, torch.Tensor) and item.size(0) == N:
                        data[key] = item.narrow(0, start, length)
                    elif isinstance(item, torch.Tensor) and item.size(0) == E and item.ndim > 1:
                        data[key] = item[edge_idx]
                    else:
                        data[key] = item

                row, col, _ = adj.coo()
                data.edge_index = torch.stack([row, col], dim=0)
                
                if split == 'test' and False:
                    node_config_bar = tqdm(range(len(data.y)))
                    node_config_bar.set_description_str(f'Config Batch-{i}, Seg-{j}')
                else:
                    node_config_bar = range(len(data.y))

                for k in node_config_bar:
                    unfold_g = Data(
                        edge_index=data.edge_index,
                        op_feats=data.op_feats,
                        op_code=data.op_code,
                        config_feats=data.config_feats_full[:, k, :], 
                        num_nodes=length
                    )
                    for k in data.keys:
                        if k not in unfold_g.keys:
                            setattr(unfold_g, k, getattr(data, k))
                    batch_seg.append(unfold_g)

        def partial_inference(batch_seg: List[Data]) -> torch.Tensor:
            nonlocal model
            batch_seg = Batch.from_data_list(batch_seg)  # (batch_size * sum(num_segments[i], * num_config,)
            batch_seg.to(torch.device(cfg.device))
            batch_seg = model.gather_input_feat(batch_seg)
        
            custom_gnn = model.model.model
            module_len = len(list(custom_gnn.children()))
            for i, module in enumerate(custom_gnn.children()):
                if i < module_len - 1:
                    batch_seg = module(batch_seg)
                if i == module_len - 1:
                    batch_seg_embed = tnn.global_max_pool(batch_seg.x, batch_seg.batch) + \
                        tnn.global_mean_pool(batch_seg.x, batch_seg.batch)
            graph_embed = batch_seg_embed / torch.norm(batch_seg_embed, dim=-1, keepdim=True)
            for i, module in enumerate(custom_gnn.children()):
                if i == module_len - 1:
                    res = module.layer_post_mp(graph_embed)
            return res
        
        res = []
        batch_graphs = cfg.train.batch_size * cfg.dataset.num_sample_config
        inference_bar = tqdm(range(0, len(batch_seg), batch_graphs))
        inference_bar.set_description_str('Partial Batch Inferece')
        for i in inference_bar:
            res.append(
                partial_inference(batch_seg[i: i + batch_graphs])
            )
        res = torch.cat(res, dim=0)
        
        true = true.to(torch.device(cfg.device))
        pred = torch.zeros([
            len(batch_list), 
            batch_num_sample, 
            1
        ]).to(
            torch.device(cfg.device)
        )
        part_cnt = 0
        for i, num_parts in enumerate(batch_num_parts):
            for _ in range(num_parts):
                for j in range(len(sampled_idx[i])):
                    pred[i, j, :] += res[part_cnt, :]
                    part_cnt += 1

        extra_stats = {}
        if 'TPUGraphs' in cfg.dataset.name:
            pred = pred.view(-1, batch_num_sample)
            true = true.view(-1, batch_num_sample)
            loss = pairwise_hinge_loss_batch(pred, true)
            _true = true.detach().to('cpu')
            _pred = pred.detach().to('cpu')
            
            cur_task = cfg.dataset.get('tpu_task', 'layout')
            for batch_i, (runtimes, indies) in enumerate(zip(_pred, sampled_idx)):
                runtimes = runtimes.cpu().tolist()
                indies = indies.cpu().tolist()
                
                graph_name = batch_list[batch_i].graph_name
                if cur_task == 'layout':
                    item_name = f"layout:{cfg.dataset.source}:{cfg.dataset.search}:{graph_name}"
                else:
                    item_name = f"tile:xla:{graph_name}"
                
                ordered = set((rt, ind) for rt, ind in zip(runtimes, indies))
                ordered = sorted(ordered)
                
                graph_configs = torch.cat([segs.config_feats.int() for segs in batch_seg], dim=0)
                same_cfgs = all((graph_configs[j] == graph_configs[0]).all() for j in range(len(graph_configs)))
                exact_inorder = all([o[1] == i for i, o in enumerate(ordered)])
                duplicated = item_name in rankings
                if same_cfgs or exact_inorder:
                    ulogger.warning(f"Weird prediction values detected! {same_cfgs}, {exact_inorder}, {batch.graph_name}")
                abnorm = duplicated or (exact_inorder and not same_cfgs)
                # HACK: xla-tile test set have some known issue, ignore it for now.
                if 'test' in split and cur_task != 'tile' and abnorm:
                    breakpoint()
                    raise RuntimeError('Weird prediction values detected!')
                
                if cur_task == 'tile':
                    ordered = ordered[:10]
                # cfg_rank_str = ";".join([str(o[1]) for o in ordered])
                # rankings[item_name] = cfg_rank_str
                rankings[item_name] = ordered
        else:
            loss, pred_score = compute_loss(pred, true)
            _true = true.detach().to('cpu')
            _pred = pred_score.detach().to('cpu')
        logger.update_stats(true=_true,
                            pred=_pred,
                            loss=loss.detach().cpu().item(),
                            lr=0, time_used=time.time() - time_start,
                            params=cfg.params,
                            dataset_name=cfg.dataset.name,
                            **extra_stats)
        time_start = time.time()
    return rankings


@register_train('custom_tpu')
def custom_train(loggers, loaders, model: TPUModel, optimizer, scheduler):
    """
    Customized training pipeline.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    # BUG: if resume from non-eval-epoch perf[i] will be empty, use this var to force eval.
    first_run_epoch = True
    start_epoch = 0
    model = model.to(cfg.device)
    
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, optimizer, scheduler, cfg.train.epoch_resume)
    if cfg.model_ckpt:
        global logger
        logger.info(f"Load model weight from: {cfg.model_ckpt}")
        checkpoint = torch.load(cfg.model_ckpt, map_location='cpu')
        model.load_state_dict(checkpoint['model_state'], strict=False)
    
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch %s', start_epoch)

    if cfg.wandb.use:
        try:
            import wandb
        except:
            raise ImportError('WandB is not installed.')
        if cfg.wandb.name == '':
            wandb_name = make_wandb_name(cfg)
        else:
            wandb_name = cfg.wandb.name
        run = wandb.init(entity=cfg.wandb.entity, project=cfg.wandb.project,
                         name=wandb_name)
        run.config.update(cfg_to_dict(cfg))

    num_splits = len(loggers)
    split_names = [loader.dataset.split_name for loader in loaders]
    # split_names = ['val', 'test']
    full_epoch_times = []
    perf = [[] for _ in range(num_splits)]
    # emb_table = History(500000000, 1)
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        start_time = time.perf_counter()
        try:
            train_epoch(loggers[0], loaders[0], model, optimizer, scheduler, model.history,
                        cfg.optim.batch_accumulation)
        except KeyboardInterrupt:
            save_ckpt(model, optimizer, scheduler, cur_epoch)
            break
        perf[0].append(loggers[0].write_epoch(cur_epoch))

        if is_eval_epoch(cur_epoch) or first_run_epoch:
            for i in range(1, num_splits):
                if i == num_splits - 1:  # HACK: skip test-set
                    perf[i].append(perf[i - 1][-1])
                    continue
                eval_epoch(loggers[i], loaders[i], model,
                           split=split_names[i])
                perf[i].append(loggers[i].write_epoch(cur_epoch))
        else:
            for i in range(1, num_splits):
                perf[i].append(perf[i][-1])

        first_run_epoch = False
        val_perf = perf[1]
        if cfg.optim.scheduler == 'reduce_on_plateau':
            scheduler.step(val_perf[-1]['loss'])
        else:
            scheduler.step()
        full_epoch_times.append(time.perf_counter() - start_time)
        # Checkpoint with regular frequency (if enabled).
        if cfg.train.enable_ckpt and is_ckpt_epoch(cur_epoch):
            save_ckpt(model, optimizer, scheduler, cur_epoch)

        if cfg.wandb.use:
            run.log(flatten_dict(perf, prefixes=split_names), step=cur_epoch)

        # Log current best stats on eval epoch.
        if is_eval_epoch(cur_epoch):
            best_epoch = np.array([vp['loss'] for vp in val_perf]).argmin()
            best_train = best_val = best_test = ""
            if cfg.metric_best != 'auto':
                # Select again based on val perf of `cfg.metric_best`.
                m = cfg.metric_best
                best_epoch = getattr(np.array([vp[m] for vp in val_perf]),
                                     cfg.metric_agg)()
                if m in perf[0][best_epoch]:
                    best_train = f"train_{m}: {perf[0][best_epoch][m]:.4f}"
                else:
                    # Note: For some datasets it is too expensive to compute
                    # the main metric on the training set.
                    best_train = f"train_{m}: {0:.4f}"
                best_val = f"val_{m}: {perf[1][best_epoch][m]:.4f}"
                best_test = f"test_{m}: {perf[2][best_epoch][m]:.4f}"

                if cfg.wandb.use:
                    bstats = {"best/epoch": best_epoch}
                    for i, s in enumerate(['train', 'val', 'test']):
                        bstats[f"best/{s}_loss"] = perf[i][best_epoch]['loss']
                        if m in perf[i][best_epoch]:
                            bstats[f"best/{s}_{m}"] = perf[i][best_epoch][m]
                            run.summary[f"best_{s}_perf"] = \
                                perf[i][best_epoch][m]
                        for x in ['hits@1', 'hits@3', 'hits@10', 'mrr']:
                            if x in perf[i][best_epoch]:
                                bstats[f"best/{s}_{x}"] = perf[i][best_epoch][x]
                    run.log(bstats, step=cur_epoch)
                    run.summary["full_epoch_time_avg"] = np.mean(full_epoch_times)
                    run.summary["full_epoch_time_sum"] = np.sum(full_epoch_times)
            # Checkpoint the best epoch params (if enabled).
            if cfg.train.enable_ckpt and cfg.train.ckpt_best and \
                    best_epoch == len(val_perf) - 1:
                src_w = model.state_dict()
                src_w.pop('history.emb')
                # save_ckpt(CheckpointWrapper(src_w), optimizer, scheduler, cur_epoch)
                ckpt_path = os.path.join(cfg.run_dir, f'best-{cur_epoch}.ckpt')
                torch.save({"model_state": src_w}, ckpt_path)
                
                if cfg.wandb.use:
                    art_name = f"{cfg.dataset.source}-{cfg.dataset.search}-{wandb.run.id}"
                    model_artifact = wandb.Artifact(
                        art_name.replace('+', '-'), 
                        type='model', 
                        metadata={
                            'epoch': cur_epoch,
                            'val_opa': val_perf[-1]['opa'],
                            'params': val_perf[-1]['params'],
                        }
                    )
                    model_artifact.add_file(ckpt_path, name=os.path.basename(ckpt_path))
                    wandb.log_artifact(model_artifact)
                
                if cfg.train.ckpt_clean:  # Delete old ckpt each time.
                    clean_ckpt()
            
            logging.info(
                f"> Epoch {cur_epoch}: took {full_epoch_times[-1]:.1f}s "
                f"(avg {np.mean(full_epoch_times):.1f}s) | "
                f"Best so far: epoch {best_epoch}\t"
                f"train_loss: {perf[0][best_epoch]['loss']:.4f} {best_train}\t"
                f"val_loss: {perf[1][best_epoch]['loss']:.4f} {best_val}\t"
                f"test_loss: {perf[2][best_epoch]['loss']:.4f} {best_test}"
            )
            if hasattr(model, 'trf_layers'):
                # Log SAN's gamma parameter values if they are trainable.
                for li, gtl in enumerate(model.trf_layers):
                    if torch.is_tensor(gtl.attention.gamma) and \
                            gtl.attention.gamma.requires_grad:
                        logging.info(f"    {gtl.__class__.__name__} {li}: "
                                     f"gamma={gtl.attention.gamma.item()}")
    logging.info(f"Avg time per epoch: {np.mean(full_epoch_times):.2f}s")
    logging.info(f"Total train loop time: {np.sum(full_epoch_times) / 3600:.2f}h")
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()
    # close wandb
    if cfg.wandb.use:
        run.finish()
        run = None

    logging.info('Task done, results saved in %s', cfg.run_dir)


@register_train('inference-tpu')
def inference_only(loggers, loaders, model: TPUModel, optimizer=None, scheduler=None):
    """
    Customized pipeline to run inference only.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    model = model.to(cfg.device)
    if not cfg.model_ckpt:
        if cfg.train.auto_resume:
            load_ckpt(model, optimizer, scheduler, cfg.train.epoch_resume)
    else:
        ulogger.info(f"Load model weight from: {cfg.model_ckpt}")
        checkpoint = torch.load(cfg.model_ckpt, map_location='cpu')
        model.load_state_dict(checkpoint['model_state'], strict=False)

    num_splits = len(loggers)
    split_names = [loader.dataset.split_name for loader in loaders]
    perf = [[] for _ in range(num_splits)]
    cur_epoch = 0

    for i in range(0, num_splits):
        if split_names[i] != 'test':
            continue
        rankings = eval_epoch(loggers[i], loaders[i], model,
                                split=split_names[i])
        
        if split_names[i] == 'test':
            df_dict = {
                'ID': list(rankings.keys()), 
                'TopConfigs': [
                    ';'.join(str(ind) for runtime, ind in ranks)
                    for ranks in rankings.values()
                ]
            }
            now = datetime.datetime.now()
            time_stamp = f"{now.year}{now.month:02}{now.day:02}_{int(now.timestamp())}"
            sub_file = os.path.join(cfg.out_dir, f'submission_{time_stamp}.csv')
            pd.DataFrame.from_dict(df_dict).to_csv(sub_file, index=False)
            ulogger.info(f"Save subission csv to: {sub_file}")

            sub_file = os.path.join(cfg.out_dir, f'submission_{time_stamp}.pt')
            torch.save(rankings, sub_file)
            ulogger.info(f"Save detail subission runtimes to: {sub_file}")
        
        perf[i].append(loggers[i].write_epoch(cur_epoch))


@register_train('valid-tpu')
def valid_once(loggers, loaders, model: TPUModel, optimizer=None, scheduler=None):
    """
    Customized pipeline to run inference only.

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: Unused, exists just for API compatibility
        scheduler: Unused, exists just for API compatibility
    """
    model = model.to(cfg.device)
    if not cfg.model_ckpt:
        if cfg.train.auto_resume:
            load_ckpt(model, optimizer, scheduler, cfg.train.epoch_resume)
    else:
        ulogger.info(f"Load model weight from: {cfg.model_ckpt}")
        checkpoint = torch.load(cfg.model_ckpt, map_location='cpu')
        model.load_state_dict(checkpoint['model_state'], strict=False)

    num_splits = len(loggers)
    split_names = [loader.dataset.split_name for loader in loaders]
    perf = [[] for _ in range(num_splits)]
    cur_epoch = 0

    for i in range(0, num_splits):
        if 'val' not in split_names[i]:
            continue
        _ = eval_epoch(loggers[i], loaders[i], model,
                                split=split_names[i])
        perf[i].append(loggers[i].write_epoch(cur_epoch))