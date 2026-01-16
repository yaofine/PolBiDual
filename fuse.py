import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple
from pathlib import Path

from module.fuse.generator import Generator
from module.fuse.discriminator import RGBDiscriminator, GradientDiscriminator


def compute_gradient(img: Tensor) -> Tensor:
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                            dtype=torch.float32, device=img.device).view(1, 1, 3, 3)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                            dtype=torch.float32, device=img.device).view(1, 1, 3, 3)

    if img.shape[1] == 3:
        img_gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
    else:
        img_gray = img

    grad_x = F.conv2d(img_gray, kernel_x, padding=1)
    grad_y = F.conv2d(img_gray, kernel_y, padding=1)

    return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)


def compute_gradient_penalty(D, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(real_samples.size(0), 1).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class Fuse:

    def __init__(self, config, mode: str = 'train'):
        self.config = config
        self.mode = mode
        self.device = torch.device(config.get('device', 'cuda:0'))

        self.generator = Generator(
            dim=config.fuse.get('dim', 64),
            depth=config.fuse.get('depth', 6)
        ).to(self.device)

        if mode == 'train':
            self.dis_rgb = RGBDiscriminator(
                dim=config.discriminator.rgb.get('dim', 64)
            ).to(self.device)
            self.dis_grad = GradientDiscriminator(
                dim=config.discriminator.gradient.get('dim', 64)
            ).to(self.device)

        self.loss_weights = {
            'structure': config.loss.get('structure', 1.0),
            'ssim': config.loss.get('ssim', 0.5),
            'adv_rgb': config.loss.get('adv_rgb', 0.1),
            'adv_grad': config.loss.get('adv_grad', 0.1)
        }

    def criterion_discriminator_rgb(self, dolp: Tensor, s0: Tensor) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            fake_rgb = self.generator(dolp, s0)

        d_real = self.dis_rgb(s0)
        d_fake = self.dis_rgb(fake_rgb.detach())

        d_loss = d_fake.mean() - d_real.mean()
        gp = compute_gradient_penalty(self.dis_rgb, s0, fake_rgb, self.device)

        return d_loss + 10 * gp, gp

    def criterion_discriminator_grad(self, dolp: Tensor, s0: Tensor) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            fake_rgb = self.generator(dolp, s0)

        real_grad = compute_gradient(s0)
        fake_grad = compute_gradient(fake_rgb)

        d_real = self.dis_grad(real_grad)
        d_fake = self.dis_grad(fake_grad.detach())

        d_loss = d_fake.mean() - d_real.mean()
        gp = compute_gradient_penalty(self.dis_grad, real_grad, fake_grad, self.device)

        return d_loss + 10 * gp, gp
def criterion_generator(self, dolp: Tensor, s0: Tensor) -> Dict[str, Tensor]:
        fake_rgb = self.generator(dolp, s0)

        loss_l1 = F.l1_loss(fake_rgb, s0)
        loss_ssim = 1 - self.ssim(fake_rgb, s0)

        d_fake_rgb = self.dis_rgb(fake_rgb)
        loss_adv_rgb = -d_fake_rgb.mean()

        fake_grad = compute_gradient(fake_rgb)
        d_fake_grad = self.dis_grad(fake_grad)
        loss_adv_grad = -d_fake_grad.mean()

        total_loss = (
            self.loss_weights['structure'] * loss_l1 +
            self.loss_weights['ssim'] * loss_ssim +
            self.loss_weights['adv_rgb'] * loss_adv_rgb +
            self.loss_weights['adv_grad'] * loss_adv_grad
        )

        return {
            'total_loss': total_loss,
            'l1': loss_l1,
            'ssim': loss_ssim,
            'adv_rgb': loss_adv_rgb,
            'adv_grad': loss_adv_grad
        }

    def ssim(self, img1: Tensor, img2: Tensor, window_size: int = 11) -> Tensor:
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu1 = F.avg_pool2d(img1, window_size, 1, window_size // 2)
        mu2 = F.avg_pool2d(img2, window_size, 1, window_size // 2)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, 1, window_size // 2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, 1, window_size // 2) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, window_size // 2) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()

    @torch.no_grad()
    def eval(self, dolp: Tensor, s0: Tensor) -> Tensor:
        self.generator.eval()
        return self.generator(dolp, s0)

    def load_ckpt(self, ckpt: dict):
        if 'generator' in ckpt:
            self.generator.load_state_dict(ckpt['generator'], strict=False)
        if self.mode == 'train':
            if 'dis_rgb' in ckpt:
                self.dis_rgb.load_state_dict(ckpt['dis_rgb'], strict=False)
            if 'dis_grad' in ckpt:
                self.dis_grad.load_state_dict(ckpt['dis_grad'], strict=False)

    def save_ckpt(self) -> dict:
        ckpt = {'generator': self.generator.state_dict()}
        if self.mode == 'train':
            ckpt['dis_rgb'] = self.dis_rgb.state_dict()
            ckpt['dis_grad'] = self.dis_grad.state_dict()
        return ckpt