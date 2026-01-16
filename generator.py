import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math

class BidirectionalCrossAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        self.dolp_to_s0_query = nn.Conv2d(channels, channels // reduction, 1)
        self.dolp_to_s0_key = nn.Conv2d(channels, channels // reduction, 1)
        self.dolp_to_s0_value = nn.Conv2d(channels, channels, 1)
        self.s0_to_dolp_query = nn.Conv2d(channels, channels // reduction, 1)
        self.s0_to_dolp_key = nn.Conv2d(channels, channels // reduction, 1)
        self.s0_to_dolp_value = nn.Conv2d(channels, channels, 1)
        self.alpha = nn.Parameter(torch.ones(1) * 0.5)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, dolp_feat, s0_feat):
        B, C, H, W = dolp_feat.shape
        if H * W > 1024:
            scale = int(math.sqrt(H * W / 1024))
            dolp_down = F.avg_pool2d(dolp_feat, scale)
            s0_down = F.avg_pool2d(s0_feat, scale)
            _, _, H_d, W_d = dolp_down.shape
        else:
            dolp_down, s0_down = dolp_feat, s0_feat
            H_d, W_d = H, W
        q_dolp = self.dolp_to_s0_query(dolp_down).view(B, -1, H_d * W_d).permute(0, 2, 1)
        k_s0 = self.dolp_to_s0_key(s0_down).view(B, -1, H_d * W_d)
        v_s0 = self.dolp_to_s0_value(s0_down).view(B, -1, H_d * W_d)
        attn_dolp_s0 = torch.bmm(q_dolp, k_s0) / math.sqrt(q_dolp.size(-1))
        attn_dolp_s0 = F.softmax(attn_dolp_s0, dim=-1)
        out_dolp = torch.bmm(v_s0, attn_dolp_s0.permute(0, 2, 1)).view(B, C, H_d, W_d)
        if H_d != H:
            out_dolp = F.interpolate(out_dolp, size=(H, W), mode='bilinear')
        q_s0 = self.s0_to_dolp_query(s0_down).view(B, -1, H_d * W_d).permute(0, 2, 1)
        k_dolp = self.s0_to_dolp_key(dolp_down).view(B, -1, H_d * W_d)
        v_dolp = self.s0_to_dolp_value(dolp_down).view(B, -1, H_d * W_d)
        attn_s0_dolp = torch.bmm(q_s0, k_dolp) / math.sqrt(q_s0.size(-1))
        attn_s0_dolp = F.softmax(attn_s0_dolp, dim=-1)
        out_s0 = torch.bmm(v_dolp, attn_s0_dolp.permute(0, 2, 1)).view(B, C, H_d, W_d)
        if H_d != H:
            out_s0 = F.interpolate(out_s0, size=(H, W), mode='bilinear')
        alpha = torch.sigmoid(self.alpha)
        aligned_dolp = self.gamma * (alpha * out_dolp + (1 - alpha) * dolp_feat) + dolp_feat
        aligned_s0 = self.gamma * ((1 - alpha) * out_s0 + alpha * s0_feat) + s0_feat
        return aligned_dolp, aligned_s0

class Generator(nn.Module):
    def __init__(self, dim=64, depth=6):
        super().__init__()
        self.dolp_encoder = nn.Sequential(
            nn.Conv2d(1, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.s0_encoder = nn.Sequential(
            nn.Conv2d(3, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.bidirectional_align = BidirectionalCrossAttention(dim)
        self.mask_generator = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim // 2, 1, 3, 1, 1),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.InstanceNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, 1, 1),
                nn.InstanceNorm2d(dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(dim, dim, 3, 1, 1),
                nn.InstanceNorm2d(dim)
            ) for _ in range(depth)
        ])
        self.rgb_head = nn.Sequential(
            nn.Conv2d(dim, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, dolp: Tensor, s0: Tensor) -> Tensor:
        dolp_feat = self.dolp_encoder(dolp)
        s0_feat = self.s0_encoder(s0)
        dolp_aligned, s0_aligned = self.bidirectional_align(dolp_feat, s0_feat)
        mask = self.mask_generator(dolp_aligned)
        dolp_weighted = dolp_aligned * mask
        s0_weighted = s0_aligned * (1 - mask)
        fused = self.fusion(torch.cat([dolp_weighted, s0_weighted], dim=1))
        for block in self.residual_blocks:
            fused = F.leaky_relu(fused + block(fused), 0.2, inplace=True)
        rgb = self.rgb_head(fused)
        rgb = (rgb + 1) / 2
        return torch.clamp(rgb, 0, 1)