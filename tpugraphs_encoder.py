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



class LitAutoEncoder(pl.LightningModule):
    def __init__(self, feat_dim: int, emb_dim: int=256, drop_prob: float=0.2):
        super().__init__()
        self.opcode_emb = nn.Embedding(128, 128, max_norm=True)
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, emb_dim * 4), 
            nn.ReLU(), 
            nn.BatchNorm1d(emb_dim * 4),
            nn.Dropout(p=drop_prob),
            
            nn.Linear(emb_dim * 4, emb_dim * 4), 
            nn.ReLU(), 
            nn.BatchNorm1d(emb_dim * 4),
            nn.Dropout(p=drop_prob),
            
            nn.Linear(emb_dim * 4, emb_dim * 2), 
            nn.ReLU(), 
            nn.BatchNorm1d(emb_dim * 2),
            nn.Dropout(p=drop_prob),
            
            nn.Linear(emb_dim * 2, emb_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2), 
            nn.ReLU(), 
            nn.BatchNorm1d(emb_dim * 2),
            nn.Dropout(p=drop_prob),
            
            nn.Linear(emb_dim * 2, emb_dim * 4), 
            nn.ReLU(), 
            nn.BatchNorm1d(emb_dim * 4),
            nn.Dropout(p=drop_prob),
            
            nn.Linear(emb_dim * 4, emb_dim * 4), 
            nn.ReLU(), 
            nn.BatchNorm1d(emb_dim * 4),
            nn.Dropout(p=drop_prob),
            
            nn.Linear(emb_dim * 4, feat_dim)
        )

    def forward(self, batch, batch_idx):
        opcode_emb = self.opcode_emb(batch['op_code'].long())
        features = batch['features']
        x = torch.cat([opcode_emb, features], dim=-1)
        
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        return x_hat, z, loss
    
    def training_step(self, batch, batch_idx):
        x_hat, z, loss = self.forward(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x_hat, z, loss = self.forward(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def collect_fn(batch: List[Dict[str, Tensor]]) -> Tensor:
    tensor_2d = defaultdict(list)
    for data in batch:
        for k, v in data.items():
            tensor_2d[k].append(v)
    tensor_2d = {
        k: torch.concat(v, dim=0)
        for k, v in tensor_2d.items()
    }
    return tensor_2d


def fourier_enc(ten: Tensor, scales=[-1, 0, 1, 2, 3, 4, 5, 6]) -> Tensor:
    """
    ten: (n, feature_dim)
    return: (n, *feature_dim)
    """
    
    def multiscale(x, scales):
        return torch.hstack([x / pow(3., i) for i in scales])
    
    return torch.hstack([
        torch.sin(multiscale(ten, scales)), 
        torch.cos(multiscale(ten, scales))
    ])


def transform_feat(op_feats: Tensor):
    op_shape = fourier_enc(op_feats[:, 21: 30])
    op_parameters = fourier_enc(op_feats[:, 30: 31])
    op_dims = fourier_enc(op_feats[:, 31: 37])
    op_win_size = fourier_enc(op_feats[:, 37: 45])
    op_win_stride = fourier_enc(op_feats[:, 45: 53])
    op_win_lowpad = fourier_enc(op_feats[:, 53: 61])
    op_win_hipad = fourier_enc(op_feats[:, 61: 69])
    op_win_dila = fourier_enc(op_feats[:, 69: 85])
    op_win_rever = op_feats[:, 85: 93]
    op_else = fourier_enc(op_feats[:, 93:])
    encoded = torch.cat(
        [
            op_feats[:, :21],
            op_shape,
            op_parameters,
            op_dims,
            op_win_size,
            op_win_stride,
            op_win_lowpad,
            op_win_hipad,
            op_win_dila,
            op_win_rever,
            op_else,
        ], 
        dim=-1
    )
    return encoded

class TPUGraphNode(Dataset):

    def __init__(self, data_dir: str, num_samples: int=128) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.npzs = glob.glob(os.path.join(data_dir, '**', '*.npz'), recursive=True)
        self.npzs = sorted(self.npzs)
        self.num_samples = num_samples
    
    def __len__(self):
        return len(self.npzs)
    
    def __getitem__(self, index) -> Tensor:
        np_file = dict(np.load(self.npzs[index]))
        # edge_index = torch.tensor(np_file["edge_index"].T)
        op_feats = torch.tensor(np_file["node_feat"])  # https://www.kaggle.com/competitions/predict-ai-model-runtime/data
        op_code = torch.tensor(np_file["node_opcode"])
        num_nodes = op_feats.size(0)
        # config_feats = torch.tensor(np_file["node_config_feat"])
        # config_feats = config_feats.view(-1, config_feats.shape[-1])
        # config_idx = torch.tensor(np_file["node_config_ids"]) 
        # runtime = torch.tensor(np_file["config_runtime"])

        sample_idx = torch.randint(0, len(op_feats), (min(num_nodes, self.num_samples),))
        op_feats = op_feats[sample_idx]
        op_code = op_code[sample_idx]

        global transform_feat
        op_feats = transform_feat(op_feats)
        
        return {
            'features': op_feats,
            'op_code': op_code,
        }


def test_dataset():
    dataset = TPUGraphNode("/home/ron/Projects/TPU-Graph/datasets/TPUGraphsNpz/raw/npz/layout/*/*/train")
    data_dict = dataset[0]
    print(data_dict)
    print(data_dict['features'].shape)
    print(data_dict['op_code'].shape)
    print(data_dict['features'][0])

    loader = DataLoader(dataset, collate_fn=collect_fn, batch_size=2)
    for batch in loader:
        print(batch)
        print(batch['features'].shape)
        print(batch['op_code'].shape)
        print(batch['features'][0])
        break
    print("len(dataset): ", len(dataset))


def train():
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.callbacks import ModelCheckpoint

    model = LitAutoEncoder(feat_dim=(1805 + 128), emb_dim=128)
    
    train_dataset = torch.utils.data.ConcatDataset([
        TPUGraphNode("/home/ron/Projects/TPU-Graph/datasets/TPUGraphsNpz/raw/npz/layout/*/*/train"),
        TPUGraphNode("/home/ron/Projects/TPU-Graph/datasets/TPUGraphsNpz/raw/npz/layout/*/*/test"),
        TPUGraphNode("/home/ron/Projects/TPU-Graph/datasets/TPUGraphsNpz/raw/npz/layout/*/*/valid"),
    ])
    val_dataset = TPUGraphNode("/home/ron/Projects/TPU-Graph/datasets/TPUGraphsNpz/raw/npz/layout/xla/random/valid")
    train_loader = DataLoader(
        train_dataset, 
        collate_fn=collect_fn, 
        batch_size=4,
        num_workers=4,
        shuffle=True
    )
    val_loader = DataLoader(val_dataset, collate_fn=collect_fn, batch_size=2)
    
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
        val_dataloaders=val_loader,
        # ckpt_path='lightning_logs/version_5/checkpoints/last.ckpt',
    )


@torch.no_grad()
def insert_node_feature(data_dir, ckpt, new_name="op_feat_enc_i"):
    model = LitAutoEncoder(feat_dim=(1805 + 128), emb_dim=128)
    model.load_state_dict(torch.load(ckpt)['state_dict'])
    model = model.to('cuda').eval()
    
    graph_files = glob.glob(data_dir)
    graph_files = sorted(graph_files)

    encoded_feats = []
    encode_file = []
    for f in tqdm(graph_files, desc='Create embedding'):
        graph = torch.load(f)
        if hasattr(graph, "op_feats"):
            data = {
                'features': transform_feat(graph.op_feats.to('cuda')),
                'op_code': graph.op_code.to('cuda'),
            }
            recon, encoded, _  = model(data, 0)
            encoded_feats.append(encoded.cpu())
            encode_file.append(f)
    
    all_feats = torch.concat(encoded_feats, dim=0)
    op_feats_mean = torch.mean(all_feats, dim=0, keepdim=True)
    op_feats_std = torch.std(all_feats, dim=0, keepdim=True)
    op_feats_std[op_feats_std < 1e-6] = 1e-6
    
    for f, feat in tqdm(zip(encode_file, encoded_feats), desc='Insert Norm Embed'):
        graph = torch.load(f)
        if hasattr(graph, "op_feats"):
            norm_feat = (feat - op_feats_mean) / op_feats_std
            setattr(graph, new_name, norm_feat)
            torch.save(graph, f)


if __name__ == '__main__':
    # test_dataset()
    # train()
    insert_node_feature(
        "datasets/TPUGraphsNpz/processed/xla_tile_data_*.pt", 
        "feat-encoder-v2.ckpt",
    )