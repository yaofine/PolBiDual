import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from pathlib import Path
from typing import Literal, List, Tuple, Dict
from torchvision.ops import box_convert

from module.detect.models.yolo import Model
from module.detect.utils.general import labels_to_class_weights, non_max_suppression
from module.detect.utils.loss import ComputeLoss
from module.detect.utils.metrics import box_iou
import numpy


class MultiScaleFeatureExtractor(nn.Module):

    def __init__(self, scales=[8, 16, 32]):
        super().__init__()
        self.scales = scales

        self.extractors = nn.ModuleDict({
            f'P{i}': nn.Sequential(
                nn.Conv2d(3, 64, 7, stride=scale, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 256, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ) for i, scale in enumerate(scales, 3)
        })

    def forward(self, imgs: Tensor) -> Dict[str, Tensor]:
        if imgs.shape[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        features = {}
        for name, extractor in self.extractors.items():
            features[name] = extractor(imgs)
        return features


class Detect:

    def __init__(self, config, mode: Literal['train', 'eval'] = 'train'):
        self.config = config
        self.mode = mode
        self.device = torch.device(config.get('device', 'cuda:0'))

        self.net = Model(config.detect.cfg, ch=3, nc=config.detect.nc).to(self.device)
        self.stride = self.net.stride

        if mode == 'train':
            self.compute_loss = ComputeLoss(self.net)
            self.feature_extractor = MultiScaleFeatureExtractor().to(self.device)
        else:
            self.net.eval()

    def train_step(self, fused_imgs: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        self.net.train()
        pred = self.net(fused_imgs)
        loss, loss_items = self.compute_loss(pred, labels)
        return loss, loss_items

    @torch.no_grad()
    def predict(self, fused_imgs: Tensor, conf_thres=0.25, iou_thres=0.45) -> List[Tensor]:
        self.net.eval()
        pred = self.net(fused_imgs)[0]
        pred = non_max_suppression(
            pred,
            conf_thres,
            iou_thres,
            classes=self.config.detect.get('classes', None),
            agnostic=self.config.detect.get('agnostic_nms', False),
            multi_label=True,
            max_det=self.config.detect.get('max_det', 300)
        )
        return pred

    def get_features(self, s0_imgs: Tensor) -> Dict[str, Tensor]:
        return self.feature_extractor(s0_imgs)
def load_ckpt(self, ckpt: dict):
        if 'detect' in ckpt:
            self.net.load_state_dict(ckpt['detect'], strict=False)
        if self.mode == 'train' and 'feature_extractor' in ckpt:
            self.feature_extractor.load_state_dict(ckpt['feature_extractor'], strict=False)

    def save_ckpt(self) -> dict:
        ckpt = {'detect': self.net.state_dict()}
        if self.mode == 'train':
            ckpt['feature_extractor'] = self.feature_extractor.state_dict()
        return ckpt

    def param_groups(self):
        from functions.get_param_groups import get_param_groups
        groups = get_param_groups(self.net)

        if self.mode == 'train':
            groups[0].extend(list(self.feature_extractor.parameters()))

        return groups

    @staticmethod
    def process_batch(detections, labels, iou_v):
        correct = torch.zeros(detections.shape[0], iou_v.shape[0], dtype=torch.bool, device=iou_v.device)
        iou = box_iou(labels[:, 1:], detections[:, :4])
        x = torch.where((iou >= iou_v[0]) & (labels[:, 0:1] == detections[:, 5]))

        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[numpy.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[numpy.unique(matches[:, 0], return_index=True)[1]]
            matches = torch.from_numpy(matches).to(iou_v.device)
            correct[matches[:, 1].long()] = matches[:, 2:3] >= iou_v
        return correct