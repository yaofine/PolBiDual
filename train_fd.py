import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
import json
import csv

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
import matplotlib
import matplotlib.pyplot as plt
import yaml

matplotlib.use('Agg')

import loader
from config import from_dict
from module.detect.utils.metrics import ap_per_class
from pipeline.detect import Detect
from pipeline.fuse import Fuse
from tools.dict_to_device import dict_to_device


class TrainFD:

    def __init__(self, config, wandb_key=None):
        logging.basicConfig(
            level='INFO',
            format='%(asctime)s | %(filename)s[line:%(lineno)d] | %(levelname)s | %(message)s'
        )

        if isinstance(config, (str, Path)):
            self.config_path = Path(config)
            config = yaml.safe_load(self.config_path.read_text())
        self.config = from_dict(config)

        self.device = torch.device(self.config.get('device', 'cuda:0'))

        self.fuse = Fuse(self.config, mode='train')
        self.detect = Detect(self.config, mode='train')

        self.train_loader = loader.get_loader(
            self.config.dataset.train,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            num_workers=self.config.train.get('num_workers', 4)
        )
        self.val_loader = loader.get_loader(
            self.config.dataset.val,
            batch_size=self.config.train.batch_size,
            shuffle=False,
            num_workers=self.config.train.get('num_workers', 4)
        )

        self.g_optimizer = torch.optim.AdamW(
            self.fuse.generator.parameters(),
            lr=float(self.config.train.lr_g),
            betas=(0.5, 0.999)
        )
        self.d_optimizer = torch.optim.AdamW(
            list(self.fuse.dis_rgb.parameters()) + list(self.fuse.dis_grad.parameters()),
            lr=float(self.config.train.lr_d),
            betas=(0.5, 0.999)
        )
        self.det_optimizer = torch.optim.SGD(
            self.detect.param_groups(),
            lr=float(self.config.train.lr_det),
            momentum=0.937,
            weight_decay=0.0005,
            nesterov=True
        )

        self.g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.g_optimizer, T_max=self.config.train.epochs
        )
        self.d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.d_optimizer, T_max=self.config.train.epochs
        )
        self.det_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.det_optimizer, T_max=self.config.train.epochs
        )

        self.work_dir = Path(self.config.train.get('work_dir', './work_dir')) / \
                        datetime.now().strftime('%Y%m%d_%H%M%S')
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.work_dir / 'checkpoints'
        self.ckpt_dir.mkdir(exist_ok=True)
        self.monitor_dir = self.work_dir / 'monitor'
        self.monitor_dir.mkdir(exist_ok=True)

        self.log_path = self.work_dir / 'train_log.csv'
        self.meta_path = self.work_dir / 'meta.txt'

        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'stage', 'g_loss', 'd_loss', 'det_loss', 'mAP50', 'mAP50-95'])

        self.best_map = 0.0
        self.best_epoch = 0
        self.start_time = time.time()
def fit(self):
        for epoch in range(self.config.train.epochs):
            self.epoch = epoch
            
            if epoch < self.config.train.get('warmup_epochs', 5):
                stage = 'pretrain'
            elif epoch < self.config.train.get('joint_epochs', 15):
                stage = 'joint'
            else:
                stage = 'finetune'

            train_metrics = self.train_one_epoch(epoch, stage)
            val_metrics = self.validate(epoch)

            self.log_epoch(epoch, stage, train_metrics, val_metrics)
            self.save_checkpoints(epoch, val_metrics)

    def train_one_epoch(self, epoch, stage):
        self.fuse.generator.train()
        self.fuse.dis_rgb.train()
        self.fuse.dis_grad.train()
        self.detect.net.train()

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch} [{stage}]')
        epoch_losses = []

        for batch_i, sample in enumerate(pbar):
            sample = dict_to_device(sample, self.device)
            dolp, s0 = sample['dolp'], sample['s0']
            labels = sample['labels']

            if stage in ['pretrain', 'joint']:
                d_loss_rgb, gp_rgb = self.fuse.criterion_discriminator_rgb(dolp, s0)
                d_loss_grad, gp_grad = self.fuse.criterion_discriminator_grad(dolp, s0)
                d_loss = d_loss_rgb + d_loss_grad

                self.d_optimizer.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                g_metrics = self.fuse.criterion_generator(dolp, s0)
                g_loss = g_metrics['total_loss']

                if stage == 'joint':
                    fused_imgs = self.fuse.generator(dolp, s0)
                    det_loss, det_loss_items = self.detect.train_step(fused_imgs, labels)
                    g_loss += det_loss * self.config.loss.get('det_weight', 0.1)

                self.g_optimizer.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()

            if stage in ['joint', 'finetune']:
                with torch.no_grad():
                    fused_imgs = self.fuse.generator(dolp, s0)
                
                det_loss, det_loss_items = self.detect.train_step(fused_imgs, labels)
                
                self.det_optimizer.zero_grad()
                det_loss.backward()
                self.det_optimizer.step()

            loss_item = {
                'g_loss': g_loss.item() if 'g_loss' in locals() else 0,
                'd_loss': d_loss.item() if 'd_loss' in locals() else 0,
                'det_loss': det_loss.item() if 'det_loss' in locals() else 0
            }
            epoch_losses.append(loss_item)
            pbar.set_postfix({k: f'{v:.4f}' for k, v in loss_item.items()})

        return self.average_losses(epoch_losses)
@torch.no_grad()
    def validate(self, epoch):
        self.fuse.generator.eval()
        self.detect.net.eval()
        
        stats = []
        val_pbar = tqdm(self.val_loader, desc=f'Evaluating Epoch {epoch}')
        
        for sample in val_pbar:
            sample = dict_to_device(sample, self.device)
            dolp, s0 = sample['dolp'], sample['s0']
            labels = sample['labels']
            
            fused_imgs = self.fuse.generator(dolp, s0)
            preds = self.detect.predict(fused_imgs)
            
            for i, pred in enumerate(preds):
                target = labels[labels[:, 0] == i]
                if pred is None:
                    if target.shape[0]:
                        stats.append((torch.zeros(0, 10), torch.Tensor(), torch.Tensor(), target[:, 1].tolist()))
                    continue
                
                if target.shape[0]:
                    correct = self.detect.process_batch(pred, target, torch.linspace(0.5, 0.95, 10).to(self.device))
                    stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), target[:, 1].tolist()))

        if len(stats):
            stats = [torch.cat(x, 0).numpy() for x in zip(*stats)]
            tp, fp, p, r, f1, ap, ap_class = ap_per_class(*stats)
            ap50, ap = ap[:, 0], ap.mean(1)
            mp, mr, map50, map_95 = p.mean(), r.mean(), ap50.mean(), ap.mean()
        else:
            map50, map_95 = 0, 0

        return {'mAP50': map50, 'mAP50-95': map_95}

    def average_losses(self, losses):
        if not losses: return {}
        return {k: np.mean([x[k] for x in losses]) for k in losses[0].keys()}

    def log_epoch(self, epoch, stage, train_metrics, val_metrics):
        g_loss = train_metrics.get('g_loss', 0)
        d_loss = train_metrics.get('d_loss', 0)
        det_loss = train_metrics.get('det_loss', 0)
        map50 = val_metrics.get('mAP50', 0)
        map_95 = val_metrics.get('mAP50-95', 0)

        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, stage, g_loss, d_loss, det_loss, map50, map_95])

        self.g_scheduler.step()
        self.d_scheduler.step()
        self.det_scheduler.step()

    def save_checkpoints(self, epoch, val_metrics):
        current_map = val_metrics.get('mAP50', 0)
        ckpt = {
            'epoch': epoch,
            'fuse': self.fuse.save_ckpt(),
            'detect': self.detect.save_ckpt(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'det_optimizer': self.det_optimizer.state_dict(),
            'mAP': current_map
        }
        
        torch.save(ckpt, self.ckpt_dir / 'last.pt')
        if current_map > self.best_map:
            self.best_map = current_map
            self.best_epoch = epoch
            torch.save(ckpt, self.ckpt_dir / 'best.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()
    
    trainer = TrainFD(args.config)
    trainer.fit()