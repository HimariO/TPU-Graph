import glob
import datetime
import os
import torch
import logging
from typing import *
from collections import defaultdict

import graphgps  # noqa, register custom modules
from graphgps.optimizer.extra_optimizers import ExtendedSchedulerConfig
from graphgps.loader.custom_loader import create_loader

import pandas as pd
import numpy as np
import lightning.pytorch as pl
from tqdm import tqdm
from torch import optim, nn, utils, Tensor
from torch_geometric.graphgym.cmd_args import parse_args
from torch_geometric.graphgym.config import (
    cfg, dump_cfg,
    set_cfg, load_cfg,
    makedirs_rm_exist
)
# from torch_geometric.graphgym.loader import create_loader
from torch_geometric.graphgym.logger import set_printing
from torch_geometric.graphgym.optim import create_optimizer, \
    create_scheduler, OptimizerConfig
from torch_geometric import seed_everything
from torch.utils.data import Dataset, DataLoader

from graphgps.finetuning import load_pretrained_model_cfg, \
    init_model_from_pretrained
from graphgps.logger import create_logger


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


def eval_opa(y_true, y_pred):
    num_preds = y_pred.shape[0]
    i_idx = torch.arange(num_preds).repeat(num_preds)
    j_idx = torch.arange(num_preds).repeat_interleave(num_preds)
    pairwise_true = y_true[i_idx] > y_true[j_idx]
    opa_indices = pairwise_true.nonzero()[0].flatten()
    opa_preds = y_pred[i_idx[opa_indices]] - y_pred[j_idx[opa_indices]]
    if len(opa_indices) > 0:
        opa_acc = float((opa_preds > 0).sum()) / opa_preds.shape[0]
    else:
        opa_acc = 0.0
    return opa_acc


class LitEncoder(pl.LightningModule):
    def __init__(self, feat_dim: int, emb_dim: int=64, drop_prob: float=0.2):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, emb_dim * 2), 
            nn.ReLU(), 
            nn.BatchNorm1d(emb_dim * 2),
            nn.Dropout(p=drop_prob),
            
            nn.Linear(emb_dim * 2, emb_dim), 
            nn.ReLU(), 
            nn.BatchNorm1d(emb_dim),
            nn.Dropout(p=drop_prob),
            
            nn.Linear(emb_dim, 1)
        )
        self.training_step_outputs = []

    def forward(self, batch, batch_idx):
        x, _ = batch
        b, nc, m, w = x.shape  # (batch, num configs, num_models, window size)
        x = x.view(b * nc, m * w)
        z = self.encoder(x)
        z = z.view(b, nc)
        return z
    
    def training_step(self, batch, batch_idx):
        z = self.forward(batch, batch_idx)
        _, y = batch
        loss = pairwise_hinge_loss_batch(z, y)
        self.log("train_loss", loss, prog_bar=True)
        opa = [eval_opa(y[i].cpu(), z[i].cpu()) for i in range(len(batch))]
        opa = sum(opa) / len(opa)
        self.log("train_opa", opa, prog_bar=True)
        self.training_step_outputs.append({'loss': loss, 'opa': opa})
        return loss
    
    def on_train_epoch_end(self) -> None:
        loss_list = [v['loss'] for v in self.training_step_outputs]
        opa_list = [v['opa'] for v in self.training_step_outputs]
        loss_mean = torch.stack(loss_list).mean()
        opa_mean = sum(opa_list) / len(opa_list)
        self.log("train_loss_epoch", loss_mean, prog_bar=False)
        self.log("train_opa_epoch", opa_mean, prog_bar=False)
        return super().on_train_epoch_end()
    
    def validation_step(self, batch, batch_idx):
        z = self.forward(batch, batch_idx)
        _, y = batch
        loss = pairwise_hinge_loss_batch(z, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class TPUGraphPred(Dataset):

    def __init__(self, predict_pts: str, num_sample: int=64, window_size=20, train=True):
        # NOTE: order of predict_pts matter!
        super().__init__()
        self.train = train
        self.num_sample = num_sample
        self.window_size = window_size

        self.logits: Dict[str, Tensor] = defaultdict(list)
        self.logits_idx: Dict[str, Tensor] = defaultdict(list)
        self.labels: Dict[str, Tensor] = {}
        self.config_idx: Dict[str, set] = defaultdict(set)
        
        for f in predict_pts:
            data = torch.load(f)
            if len(self.labels) == 0:
                for name, order_rt in data['labels'].items():
                    og_order = sorted([(i, r) for r, i in order_rt])
                    self.labels[name] = [r for i, r in og_order]
                    self.config_idx[name].update([i for i, r in og_order])
            else:
                for name, order_rt in data['labels'].items():
                    dif = self.config_idx[name].symmetric_difference(set(i for r, i in order_rt))
                    assert not dif, f"{dif}"

            for name, order_pred in data['rankings'].items():
                self.logits[name].append(
                    [0] * self.window_size + [v for v, i in order_pred] + [0] * self.window_size)
                self.logits_idx[name].append(
                    [-1] * self.window_size + [i for v, i in order_pred] + [-1] * self.window_size)
        
        self.graph_names = list(self.logits.keys())
        for name in self.graph_names:
            self.logits[name] = torch.tensor(self.logits[name])
            self.logits_idx[name] = torch.tensor(self.logits_idx[name])
            self.labels[name] = torch.tensor(self.labels[name])
    
    def __len__(self):
        if self.train:
            return len(self.graph_names) * 100
        else:
            return len(self.graph_names)
    
    def __getitem__(self, index) -> Tensor:
        name = self.graph_names[index % len(self.graph_names)]
        n = self.labels[name].size(0)
        logits = self.logits[name]
        if self.train:
            cfg_idx = torch.randperm(n)[:self.num_sample]
        else:
            cfg_idx = torch.arange(0, n)

        inputs = []
        for i in cfg_idx:
            item_and_knn = []
            for j, per_model in enumerate(self.logits_idx[name]):
                mask = int((per_model == i).nonzero()[0])
                select = torch.arange(
                    mask - self.window_size,
                    mask + self.window_size + 1,
                )
                if len(select) < self.window_size * 2 + 1:
                    print(select)
                    breakpoint()
                item_and_knn.append(logits[j][select])
            inputs.append(torch.stack(item_and_knn, dim=0))
        
        inputs = torch.stack(inputs, dim=0)
        true_runtime = self.labels[name][cfg_idx]
        return inputs, true_runtime


def test_dataset():
    dataset = TPUGraphPred([
        "./tests/xla-default-fullenc/tpu-pe-fullenc/valid_20231017_1697492282.pt",
        "./tests/xla-default-fullenc-khop/tpu-pe-fullenc-khop/valid_20231017_1697507211.pt",
    ])
    data_dict = dataset[0]
    print(data_dict)

    loader = DataLoader(dataset, batch_size=2)
    for batch in loader:
        print(batch[0].shape)
        print(batch[1].shape)
        break
    print("len(dataset): ", len(dataset))


def train():
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.callbacks import ModelCheckpoint
    pred_files = [
        "./tests/xla-random-extra-v2-full/xla-rand-extra-v2-full/valid_20231110_1699607932.pt",
        "./tests/xla-random-extra-v2-full/xla-rand-extra-v2-full/valid_20231111_1699644886.pt",
    ]
    model = LitEncoder(feat_dim=len(pred_files) * 41, emb_dim=64)
    
    train_dataset = TPUGraphPred(pred_files)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=4,
        num_workers=4,
        shuffle=True
    )
    
    ckpt_callback = ModelCheckpoint(
        save_last=True,
        every_n_train_steps=1000,
        save_on_train_epoch_end=True
    )
    trainer = pl.Trainer(
        precision=16,
        accelerator='gpu',
        check_val_every_n_epoch=2,
        callbacks=[ckpt_callback],
    )
    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
        # ckpt_path='lightning_logs/version_5/checkpoints/last.ckpt',
    )


@torch.no_grad()
def ensemble(pred_files: List[str], ckpt: str, out_csv: str):
    model = LitEncoder.load_from_checkpoint(ckpt)
    model = model.to('cuda').eval()
    
    test_set = TPUGraphPred(pred_files, train=False)
    result = {}
    for i in range(len(test_set)):
        name = test_set.graph_names[i]
        models_pred, _ = test_set[i]

        models_pred = torch.unsqueeze(models_pred.to('cuda'), dim=0)
        y = model.forward((models_pred, None), 0).cpu()[0]
        merged = []
        for config_id, score in enumerate(y):
            merged.append((float(score), config_id))
        merged = sorted(merged)
        new_rank = [str(i) for s, i in merged]
        result[name] = ";".join(new_rank)
    
    df_dict = {
        'ID': [], 
        'TopConfigs': [],
    }
    for name, rank_str in result.items():
        df_dict['ID'].append(name)
        df_dict['TopConfigs'].append(rank_str)
    pd.DataFrame.from_dict(df_dict).to_csv(out_csv, index=False)


if __name__ == '__main__':
    from loguru import logger
    with logger.catch(reraise=True):
        # test_dataset()
        # train()
        ensemble(
            [
                "tests/xla-random-extra-v2-full/xla-rand-extra-v2-full/test_20231109_1699505166.pt",
                "tests/xla-random-extra-v2-full/xla-rand-extra-v2-full/test_20231111_1699644416.pt"
            ],
            "lightning_logs/version_8/checkpoints/epoch=85-step=15000.ckpt",
            "xla_rand_meta.csv",
        )