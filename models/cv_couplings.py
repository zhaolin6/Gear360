'''This Code is based on the FrEIA Framework, source: https://github.com/VLL-HD/FrEIA
It is a assembly of the necessary modules/functions from FrEIA that are needed for our purposes.'''
from math import exp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange
import warnings
VERBOSE = False

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.checkpoint import checkpoint
from spatial_correlation_sampler import SpatialCorrelationSampler as Correlation


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class Conv_relu(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size, stride, padding, has_relu=True, efficient=False):
        super(Conv_relu, self).__init__()
        self.has_relu = has_relu
        self.efficient = efficient
        self.conv = nn.Conv2d(in_chl, out_chl, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        def _func_factory(conv, relu, has_relu):
            def func(x):
                x = conv(x)
                if has_relu:
                    x = relu(x)
                return x
            return func

        func = _func_factory(self.conv, self.relu, self.has_relu)
        if self.efficient:
            return checkpoint(func, x)
        return func(x)


try:
    from spatial_correlation_sampler import SpatialCorrelationSampler as Correlation
except Exception:
    class Correlation(nn.Module):
        def __init__(self, kernel_size, patch_size, stride=1, padding=0, dilation=1, dilation_patch=1):
            super().__init__()
            assert stride == 1 and dilation == 1 and dilation_patch == 1, "Fallback Correlation only supports stride=1 and dilation=1"
            assert patch_size % 2 == 1, "patch_size must be odd"
            self.kernel_size = kernel_size
            self.patch_size = patch_size
            self.padding = padding
            self.r = patch_size // 2

        @staticmethod
        def _shift2d(x, dy, dx):
            B, C, H, W = x.shape
            pad_l = max(-dx, 0)
            pad_r = max(dx, 0)
            pad_t = max(-dy, 0)
            pad_b = max(dy, 0)
            xp = F.pad(x, (pad_l, pad_r, pad_t, pad_b))
            xs = pad_l + dx
            ys = pad_t + dy
            return xp[:, :, ys:ys + H, xs:xs + W]

        def forward(self, input1, input2):
            B, C, H, W = input1.shape
            device = input1.device
            ones = torch.ones(1, 1, self.kernel_size, self.kernel_size, device=device, dtype=input1.dtype)

            out_rows = []
            for dy in range(-self.r, self.r + 1):
                row = []
                for dx in range(-self.r, self.r + 1):
                    shifted = self._shift2d(input2, dy, dx)
                    prod = (input1 * shifted).sum(dim=1, keepdim=True)   # [B,1,H,W]
                    corr = F.conv2d(prod, ones, padding=self.padding)     # [B,1,H,W]
                    row.append(corr)
                row = torch.cat(row, dim=1)                              # [B,patch,H,W]
                out_rows.append(row)
            out = torch.stack(out_rows, dim=1)                           # [B,patch,patch,H,W]
            return out.contiguous()


class AggregateL3Only(nn.Module):
    def __init__(self, nf=153, nbr=4, n_group=1,
                 k3=3, patch3=7, cor_ksize=3,
                 adding_avg=False, to_enhance=True,
                 use_residual=True,
                 attention_dropout=0.1,
                 feature_norm='group'):
        super().__init__()
        self.nbr = nbr
        self.g = n_group
        self.k3 = k3
        self.patch3 = patch3
        self.cor_k = cor_ksize
        self.adding_avg = adding_avg
        self.to_enhance = to_enhance
        self.use_residual = use_residual
        self.attention_dropout = attention_dropout

        assert self.k3 == self.cor_k, f"k3 must equal cor_ksize for shape compatibility, got k3={k3}, cor_ksize={cor_ksize}"

        self.L3_conv1 = Conv_relu(nf * 2, nf, 3, 1, 1, has_relu=True)
        self.L3_conv2 = Conv_relu(nf, nf, 3, 1, 1, has_relu=True)
        self.L3_conv3 = Conv_relu(nf, nf, (7, 1), 1, (3, 0), has_relu=True)
        self.L3_conv4 = Conv_relu(nf, nf, (1, 7), 1, (0, 3), has_relu=True)
        self.L3_mask = Conv_relu(nf, self.k3 ** 2, self.k3, 1, (self.k3 - 1) // 2, has_relu=False)

        self.L3_avg1 = Conv_relu(3, nf, 3, 1, 1, has_relu=True)
        self.L3_avg2 = Conv_relu(nf, self.k3 ** 2, 1, 1, 0)

        in_ch = nf * (self.nbr + (1 if self.adding_avg else 0))
        self.L3_nn_conv = Conv_relu(in_ch, nf, 3, 1, 1, has_relu=True)

        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if self.use_residual:
            self.residual_proj = nn.Conv2d(nf, nf, 1, 1, 0, bias=False)
        self.dropout = nn.Dropout2d(attention_dropout)

        self.padding = (self.cor_k - 1) // 2
        self.pad_size = self.padding + (self.patch3 - 1) // 2
        self.add_num = 2 * self.pad_size - self.cor_k + 1

        self.L3_corr = Correlation(kernel_size=self.cor_k, patch_size=self.patch3,
                                   stride=1, padding=self.padding, dilation=1, dilation_patch=1)

        if feature_norm == 'layer':
            self.norm = nn.GroupNorm(nf, nf)
        elif feature_norm == 'group':
            self.norm = nn.GroupNorm(min(32, nf), nf)
        elif feature_norm == 'batch':
            self.norm = nn.BatchNorm2d(nf)
        else:
            self.norm = nn.Identity()

    def forward(self, nbr_L3, ref_L3, recur_L3=None):
        B, C, H, W = nbr_L3.shape
        device = nbr_L3.device

        L3_w = torch.cat([nbr_L3, ref_L3], dim=1)  # [B,2C,H,W]
        L3_w = self.L3_conv4(self.L3_conv3(self.L3_conv2(self.L3_conv1(L3_w))))  # [B,C,H,W]
        L3_mask = self.L3_mask(L3_w).view(B, 1, self.k3 ** 2, H, W)      # [B,1,k3^2,H,W]

        # 2) correlation + topk
        ref_n = F.normalize(ref_L3, dim=1)
        nbr_n = F.normalize(nbr_L3, dim=1)
        L3_corr = self.L3_corr(ref_n, nbr_n).view(B, -1, H, W)  # [B,patch^2,H,W]

        _, corr_ind = torch.topk(L3_corr, self.nbr, dim=1)  # [B,nbr,H,W]
        corr_ind = corr_ind.permute(0, 2, 3, 1).reshape(B, H * W * self.nbr)

        ind_row_add = corr_ind // self.patch3 * (W + self.add_num)
        ind_col_add = corr_ind % self.patch3
        corr_ind = ind_row_add + ind_col_add

        y = torch.arange(H, device=device).repeat_interleave(W)
        x = torch.arange(W, device=device).repeat(H)
        lt_ind = y * (W + self.add_num) + x
        lt_ind = lt_ind.repeat_interleave(self.nbr).long().unsqueeze(0)
        corr_ind = (corr_ind + lt_ind).view(-1)

        nbr_unf = F.unfold(nbr_L3, self.cor_k, dilation=1, padding=self.pad_size, stride=1)
        ind_B = torch.arange(B, dtype=torch.long, device=device).repeat_interleave(H * W * self.nbr)
        L3 = nbr_unf[ind_B, :, corr_ind].view(B * H * W, self.nbr * C, self.cor_k, self.cor_k)
        # L3: [B*H*W, nbr*C, cor_k, cor_k]

        L3_corr_mask = None
        if self.to_enhance:
            L3_corr_mask = L3[:, 0:C, :, :]
            L3_corr_mask = L3_corr_mask.view(B, H, W, C, self.cor_k ** 2).permute(0, 3, 4, 1, 2)
            L3_corr_mask = L3_corr_mask.view(B, 1, C, self.cor_k ** 2, H, W)

        if self.adding_avg:
            if recur_L3 is None:
                raise ValueError("recur_L3 [B,3,H,W] is required when adding_avg=True")

            m = min(4, self.nbr)
            L3_avg = 0.0
            for j in range(m):
                L3_avg = L3_avg + L3[:, j * C:(j + 1) * C, :, :]
            L3_avg = L3_avg / float(m)

            L3_avg_mask = self.L3_avg2(self.L3_avg1(recur_L3))  # [B, 9, H, W]
            L3_avg_mask = L3_avg_mask.permute(0, 2, 3, 1).reshape(B * H * W, 1, self.cor_k, self.cor_k)
            L3_avg = L3_avg * L3_avg_mask
            L3 = torch.cat([L3, L3_avg], dim=1)  # [B*H*W, (nbr+1)*C, cor_k, cor_k]

        L3 = self.L3_nn_conv(L3)

        L3 = L3.view(B, H, W, C, self.cor_k ** 2).permute(0, 3, 4, 1, 2)  # [B,C,k3^2,H,W]
        L3 = L3.view(B, C, self.cor_k ** 2, H, W)       # [B,C,k3^2,H,W]

        # L3: [B, C, k3^2, H, W], L3_mask: [B, 1, k3^2, H, W]
        L3_aggregated = (L3 * L3_mask).sum(dim=2).view(B, C, H, W)  # [B,C,H,W]

        L3_aggregated = self.norm(L3_aggregated)
        L3_final = self.relu(L3_aggregated)

        if self.use_residual:
            residual = self.residual_proj(nbr_L3 + ref_L3)
            L3_final = L3_final + residual

        L3_final = self.dropout(L3_final)

        return L3_final, L3_corr_mask


class Fusion(nn.Module):
    def __init__(self, nf, n_view=3,
                 fusion_type='attention',
                 temperature=1.0):
        super().__init__()
        self.n_view = n_view
        self.fusion_type = fusion_type
        self.temperature = temperature

        if fusion_type == 'attention':
            self.ref_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.nbr_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
            self.fuse_conv = nn.Conv2d(nf * n_view, nf * n_view, 1, 1, bias=True)
            self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif fusion_type == 'weighted_sum':
            self.weights = nn.Parameter(torch.ones(n_view) / n_view)
        elif fusion_type == 'concat':
            self.reduce_conv = nn.Conv2d(nf * n_view, nf, 1, 1, 0, bias=True)
            self.norm = nn.GroupNorm(min(32, nf), nf)

    def forward(self, x):
        B, N, C, H, W = x.size()

        if self.fusion_type == 'attention':
            emb_ref = self.ref_conv(x[:, 0].clone())                      # [B,C,H,W]
            emb = self.nbr_conv(x.view(-1, C, H, W)).view(B, N, C, H, W)       # [B,N,C,H,W]

            cor_l = []
            for i in range(N):
                cor = torch.sum(emb[:, i] * emb_ref, dim=1, keepdim=True)      # [B,1,H,W]
                cor_l.append(cor)

            cor_prob = torch.sigmoid(torch.cat(cor_l, dim=1))                  # [B,N,1,H,W]
            cor_prob = cor_prob.unsqueeze(2).repeat(1, 1, C, 1, 1).view(B, -1, H, W)
            aggr_fea = x.view(B, -1, H, W) * cor_prob
            out = self.lrelu(self.fuse_conv(aggr_fea)).view(B, N, -1, H, W)
        elif self.fusion_type == 'weighted_sum':
            weights = F.softmax(self.weights / self.temperature, dim=0)
            out = torch.zeros_like(x)
            for i in range(N):
                out[:, i] = (x[:, i].permute(0, 2, 3, 1) * weights[i]).permute(0, 3, 1, 2)
        elif self.fusion_type == 'concat':
            x_concat = x.view(B, N * C, H, W)  # [B, N*C, H, W]
            x_reduced = self.reduce_conv(x_concat)  # [B, C, H, W]
            x_reduced = self.norm(x_reduced)
            out = x_reduced.unsqueeze(1).expand(-1, N, -1, -1, -1)  # [B, N, C, H, W]

        return out



class AlignFusionL3Only_Core(nn.Module):
    def __init__(self, nf=153, n_view=3, nbr=4, n_group=1,
                 k3=3, patch3=7, cor_ksize=3,
                 adding_avg=False, to_enhance=True,
                 use_residual=True,
                 attention_dropout=0.1,
                 feature_norm='group'):
        super().__init__()
        self.n_view = n_view
        self.center = 0

        self.aggr = AggregateL3Only(
            nf=nf, nbr=nbr, n_group=n_group,
            k3=k3, patch3=patch3, cor_ksize=cor_ksize,
            adding_avg=adding_avg, to_enhance=to_enhance,
            use_residual=use_residual,
            attention_dropout=attention_dropout,
            feature_norm=feature_norm
        )
        self.fuse = Fusion(nf=nf, n_view=n_view)

    def forward(self, L3, recur_L3=None):
        B, N, C, H, W = L3.shape
        assert N == self.n_view, f"N={N} does not match n_view={self.n_view}"

        ref = L3[:, self.center].clone()  # [B,C,H,W]

        aligned_list = []
        corr_mask_list = []

        for i in range(self.n_view):
            nbr = L3[:, i].clone()
            ali, cm = self.aggr(nbr, ref, recur_L3=recur_L3)
            aligned_list.append(ali)
            corr_mask_list.append(cm)

        aligned = torch.stack(aligned_list, dim=1)    # [B,N,C,H,W]
        out_L3 = self.fuse(aligned)                   # [B,N,C,H,W]

        corr_mask_L3 = None if corr_mask_list[0] is None else torch.stack(corr_mask_list, dim=1)
        return out_L3, corr_mask_L3


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class CrossBlockMultiViewAttention(nn.Module):

    def __init__(self, d_model, n_win=4, auto_pad=True, percentile=0.8):
        super(CrossBlockMultiViewAttention, self).__init__()
        self.d_model = d_model
        self.auto_pad = auto_pad
        self.n_win = n_win
        self.percentile = percentile
        self.dropout = nn.Dropout(0.2)

    def forward(self, cv_feature, mv_feature):
        if self.auto_pad:
            B, H_in, W_in, C = cv_feature.size()
            pad_l = pad_t = 0
            pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            cv_feature_p = F.pad(cv_feature, (0, 0, pad_l, pad_r, pad_t, pad_b))
            mv_feature = F.pad(mv_feature, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, H, W, _ = cv_feature_p.size()
        else:
            B, H, W, C = cv_feature.size()
            cv_feature_p = cv_feature
            assert H % self.n_win == 0 and W % self.n_win == 0

        cv_patch = rearrange(cv_feature_p, "b (j h) (i w) c -> b (j i) h w c",
                             j=self.n_win, i=self.n_win)  # [B, P2, bh, bw, C]
        mv_patch = rearrange(mv_feature, "b v (j h) (i w) c -> b v (j i) h w c",
                             j=self.n_win, i=self.n_win)  # [B, V, P2, bh, bw, C]

        q_win = cv_patch.mean(dim=(2, 3))  # [B, P2, C]
        k_win = mv_patch.mean(dim=(3, 4))  # [B, V, P2, C]

        q_win = F.normalize(q_win, p=2, dim=-1)
        k_win = F.normalize(k_win, p=2, dim=-1)

        sim = torch.einsum("bpc,bvqc->bvpq", q_win, k_win)  # [B, V, P2, P2]

        adaptive_threshold = torch.quantile(sim, self.percentile, dim=-1, keepdim=True)  # [B, V, 1, 1]
        adaptive_threshold = self.dropout(adaptive_threshold)
        adaptive_threshold = adaptive_threshold.expand_as(sim)

        mask = sim >= adaptive_threshold  # [B, V, P2, P2]

        max_sim_per_query = sim.max(dim=-1, keepdim=True)[0]  # [B, V, P2, 1]
        mask = mask | (sim >= max_sim_per_query - 1e-6)

        sim_masked = sim.masked_fill(~mask, float("-inf"))
        w_block = F.softmax(sim_masked, dim=-1)  # [B, V, P2, P2]
        w_block = w_block.masked_fill(~mask, 0.0)

        fused_mv = torch.einsum("bvpq,bvqhwc->bvphwc", w_block, mv_patch)  # [B, V, P2, bh, bw, C]

        q_pix = rearrange(cv_patch, "b p h w c -> b p (h w) c")  # [B, P2, L, C]
        k_pix = rearrange(fused_mv, "b v p h w c -> b v p (h w) c")  # [B, V, P2, L, C]

        outs = []
        for v in range(k_pix.size(1)):
            kv_v = k_pix[:, v]  # [B, P2, L, C]
            attn = torch.matmul(q_pix, kv_v.transpose(-2, -1)) / (C ** 0.5)  # [B, P2, L, L]
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, kv_v)  # [B, P2, L, C]
            outs.append(out)

        out = torch.stack(outs, dim=0).mean(dim=0)  # [B, P2, L, C]

        out = rearrange(out, "b p (h w) c -> b p h w c", h=H // self.n_win, w=W // self.n_win)
        out = rearrange(out, "b (j i) h w c -> b (j h) (i w) c", j=self.n_win, i=self.n_win)

        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :H_in, :W_in, :].contiguous()

        return out



class CBST(nn.Module):
    '''ResNet transformation with improved cross-block attention'''

    def __init__(self, in_channels, channels, channels_hidden=512,
                 stride=None, kernel_size=3, last_kernel_size=1, leaky_slope=0.1,
                 batch_norm=False, block_no=0, use_gamma=True):
        super(CBST, self).__init__()

        if not channels_hidden:
            channels_hidden = channels

        pad = kernel_size // 2
        self.leaky_slope = leaky_slope
        self.use_gamma = use_gamma
        pad_mode = "replicate"

        self.gamma0 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.conv_scale0_0 = nn.Conv2d(in_channels, channels_hidden, kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale1_0 = nn.Conv2d(in_channels, channels_hidden, kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale2_0 = nn.Conv2d(in_channels, channels_hidden, kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)

        self.conv_scale0_1 = nn.Conv2d(channels_hidden * 1, channels, kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale1_1 = nn.Conv2d(channels_hidden * 1, channels, kernel_size=kernel_size, padding=pad * 1,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale2_1 = nn.Conv2d(channels_hidden * 1, channels, kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)

        self.cbmva0 = CrossBlockMultiViewAttention(d_model=channels_hidden, n_win=4)
        self.cbmva1 = CrossBlockMultiViewAttention(d_model=channels_hidden, n_win=4)
        self.cbmva2 = CrossBlockMultiViewAttention(d_model=channels_hidden, n_win=4)

        self.norm0 = nn.LayerNorm(channels_hidden, eps=1e-6)
        self.norm1 = nn.LayerNorm(channels_hidden, eps=1e-6)
        self.norm2 = nn.LayerNorm(channels_hidden, eps=1e-6)

        self.drop_rate = 0.1
        self.drop_path = DropPath(self.drop_rate) if self.drop_rate > 0. else nn.Identity()

        self.mlp0 = nn.Sequential(
            nn.Linear(channels_hidden, channels_hidden * 4),
            nn.ReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(channels_hidden * 4, channels_hidden),
            nn.Dropout(self.drop_rate)
        )
        self.mlp1 = nn.Sequential(
            nn.Linear(channels_hidden, channels_hidden * 4),
            nn.ReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(channels_hidden * 4, channels_hidden),
            nn.Dropout(self.drop_rate)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(channels_hidden, channels_hidden * 4),
            nn.ReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(channels_hidden * 4, channels_hidden),
            nn.Dropout(self.drop_rate)
        )

        self.lr = nn.ReLU()

    def forward(self, x0, x1, x2):
        out0 = self.conv_scale0_0(x0)
        out1 = self.conv_scale1_0(x1)
        out2 = self.conv_scale2_0(x2)

        y0 = self.lr(out0)
        y1 = self.lr(out1)
        y2 = self.lr(out2)

        y0_perm = y0.permute(0, 2, 3, 1).contiguous()
        y1_perm = y1.permute(0, 2, 3, 1).contiguous()
        y2_perm = y2.permute(0, 2, 3, 1).contiguous()

        mv_feature_0 = torch.stack([y1_perm, y2_perm], dim=1)  # [B, 2, H, W, C]
        attn_out0 = self.norm0(y0_perm + self.drop_path(self.cbmva0(y0_perm, mv_feature_0)))
        fused0 = self.norm0(attn_out0 + self.mlp0(attn_out0))
        fused0 = fused0.permute(0, 3, 1, 2).contiguous()

        mv_feature_1 = torch.stack([y0_perm, y2_perm], dim=1)  # [B, 2, H, W, C]
        attn_out1 = self.norm1(y1_perm + self.drop_path(self.cbmva1(y1_perm, mv_feature_1)))
        fused1 = self.norm1(attn_out1 + self.mlp1(attn_out1))
        fused1 = fused1.permute(0, 3, 1, 2).contiguous()

        mv_feature_2 = torch.stack([y0_perm, y1_perm], dim=1)  # [B, 2, H, W, C]
        attn_out2 = self.norm2(y2_perm + self.drop_path(self.cbmva2(y2_perm, mv_feature_2)))
        fused2 = self.norm2(attn_out2 + self.mlp2(attn_out2))
        fused2 = fused2.permute(0, 3, 1, 2).contiguous()

        out0 = self.conv_scale0_1(fused0)
        out1 = self.conv_scale1_1(fused1)
        out2 = self.conv_scale2_1(fused2)

        if self.use_gamma:
            out0 = out0 * self.gamma0
            out1 = out1 * self.gamma1
            out2 = out2 * self.gamma2

        return out0, out1, out2

class NaiveCrossConvolutions(nn.Module):
    '''ResNet transformation, not itself reversible, just used below'''

    def __init__(self, in_channels, channels, channels_hidden=512,
                 stride=None, kernel_size=3, last_kernel_size=1, leaky_slope=0.1,
                 batch_norm=False, block_no=0, use_gamma=True):
        super(NaiveCrossConvolutions, self).__init__()
        if stride:
            warnings.warn("Stride doesn't do anything, the argument should be "
                          "removed", DeprecationWarning)
        if not channels_hidden:
            channels_hidden = channels

        pad = kernel_size // 2
        self.leaky_slope = leaky_slope
        self.use_gamma = use_gamma
        pad_mode = 'zeros'

        self.gamma0 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.conv_scale0_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale1_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale2_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)

        self.conv_scale0_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale1_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad * 1,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale2_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)

        self.up_conv10 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.up_conv21 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)

        self.down_conv01 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        self.down_conv12 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)

        self.lr = nn.ReLU() # nn.LeakyReLU(self.leaky_slope)

    def forward(self, x0, x1, x2):
        # x0 is top view, x1 and x2 are side views
        out0 = self.conv_scale0_0(x0)
        out1 = self.conv_scale1_0(x1)
        out2 = self.conv_scale2_0(x2)

        y0 = self.lr(out0)
        y1 = self.lr(out1)
        y2 = self.lr(out2)

        out0 = self.conv_scale0_1(y0)
        out1 = self.conv_scale1_1(y1)
        out2 = self.conv_scale2_1(y2)

        y1_up = self.up_conv10(y1)
        y2_up = self.up_conv21(y2)

        y0_down = self.down_conv01(y0)
        y1_down = self.down_conv12(y1)

        out0 = out0 + 0.5 * (y1_up + y2_up)
        out1 = out1 + y0_down + y2_up
        out2 = out2 + y1_down

        if self.use_gamma:
            out0 = out0 * self.gamma0
            out1 = out1 * self.gamma1
            out2 = out2 * self.gamma2
        return out0, out1, out2

class NeighboringCrossConvolutions(nn.Module):
    '''ResNet transformation, not itself reversible, just used below'''

    def __init__(self, in_channels, channels, channels_hidden=512,
                 stride=None, kernel_size=3, last_kernel_size=1, leaky_slope=0.1,
                 batch_norm=False, block_no=0, use_gamma=True):

        super(NeighboringCrossConvolutions, self).__init__()

        if stride:
            warnings.warn("Stride doesn't do anything, the argument should be "
                          "removed", DeprecationWarning)

        if not channels_hidden:
            channels_hidden = channels

        pad = kernel_size // 2
        self.leaky_slope = leaky_slope
        self.use_gamma = use_gamma
        pad_mode = "replicate"

        self.gamma0 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.conv_scale0_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale1_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale2_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)

        self.conv_scale0_1 = nn.Conv2d(channels_hidden * 1, channels,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale1_1 = nn.Conv2d(channels_hidden * 1, channels,
                                       kernel_size=kernel_size, padding=pad * 1,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale2_1 = nn.Conv2d(channels_hidden * 1, channels,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)

        self.cross_conv12 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)
        self.cross_conv21 = nn.Conv2d(channels_hidden, channels,
                                   kernel_size=kernel_size, padding=pad, bias=True, padding_mode=pad_mode)

        self.topview_up0 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        self.topview_up1 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)

        self.topview_down0 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)
        self.topview_down1 = nn.Conv2d(channels_hidden, channels,
                                     kernel_size=kernel_size, padding=pad,
                                     bias=not batch_norm, stride=1, padding_mode=pad_mode, dilation=1)

        self.lr = nn.ReLU()

    def forward(self, x0, x1, x2):
        # x0 is top view, others are side views
        out0 = self.conv_scale0_0(x0)
        out1 = self.conv_scale1_0(x1)
        out2 = self.conv_scale2_0(x2)

        y0 = self.lr(out0)
        y1 = self.lr(out1)
        y2 = self.lr(out2)

        out0 = self.conv_scale0_1(y0)
        out1 = self.conv_scale1_1(y1)
        out2 = self.conv_scale2_1(y2)

        top1_up = self.topview_up0(y0)
        top2_up = self.topview_up1(y0)

        top1_down = self.topview_down0(y1)
        top2_down = self.topview_down1(y2)

        y21 = self.cross_conv21(y2)
        y12 = self.cross_conv12(y1)

        out0 = out0 + 0.5 * (top1_down + top2_down)

        out1 = out1 + 0.5 * (top1_up + y21)

        out2 = out2 + 0.5 * (top2_up + y12)

        if self.use_gamma:
            out0 = out0 * self.gamma0
            out1 = out1 * self.gamma1
            out2 = out2 * self.gamma2
        return out0, out1, out2



class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, channels_out, kernel_size, pad, n_heads=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)
        self.conv = nn.Conv2d(embed_dim, channels_out,  #
                            kernel_size=kernel_size, padding=pad,
                            padding_mode="replicate", dilation=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.ReLU()

    def forward(self, x, k):
        B, N, H, W = x.shape

        x = x.view((x.shape[0], x.shape[1], -1)).permute(0, 2, 1)
        k = k.view((k.shape[0], k.shape[1], -1)).permute(0, 2, 1)

        out, _ = self.att(k, k, k)
        out = self.norm(self.act(out))
        out = out.permute(0, 2, 1).view((B, N, H, W))
        out = self.conv(out)
        return out

class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim, channels_out, kernel_size, pad, n_heads=4, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, batch_first=True)
        self.conv = nn.Conv2d(embed_dim, channels_out,  #
                            kernel_size=kernel_size, padding=pad,
                            padding_mode="replicate", dilation=1)
        self.norm = nn.LayerNorm(embed_dim)
        self.act = nn.ReLU()
        
    def forward(self, x, k):
        B, N, H, W = x.shape
        x = x.view((x.shape[0], x.shape[1], -1)).permute(0, 2, 1)
        k = k.view((k.shape[0], k.shape[1], -1)).permute(0, 2, 1)
        
        out, _ = self.att(x, k, k)
        out = self.norm(self.act(out))
        out = out.permute(0, 2, 1).view((B, N, H, W))
        out = self.conv(out)
        return out

class NeighboringCrossAttention(nn.Module):
    '''ResNet transformation, not itself reversible, just used below'''

    def __init__(self, in_channels, channels, channels_hidden=512,
                 stride=None, kernel_size=3, last_kernel_size=1, leaky_slope=0.1,
                 batch_norm=False, block_no=0, use_gamma=True):
        super(NeighboringCrossAttention, self).__init__()
        if stride:
            warnings.warn("Stride doesn't do anything, the argument should be "
                          "removed", DeprecationWarning)
        if not channels_hidden:
            channels_hidden = channels

        pad = kernel_size // 2
        self.leaky_slope = leaky_slope
        self.use_gamma = use_gamma
        pad_mode = "replicate"

        self.gamma0 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.conv_scale0_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale1_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale2_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)

        self.conv_scale0_1 = nn.Conv2d(channels_hidden * 1, channels,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale1_1 = nn.Conv2d(channels_hidden * 1, channels,
                                       kernel_size=kernel_size, padding=pad * 1,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale2_1 = nn.Conv2d(channels_hidden * 1, channels,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)

        self.cross_conv12 = CrossAttentionBlock(channels_hidden, channels, kernel_size, pad)
        self.cross_conv21 = CrossAttentionBlock(channels_hidden, channels, kernel_size, pad)

        self.topview_up0 = CrossAttentionBlock(channels_hidden, channels, kernel_size, pad)
        self.topview_up1 = CrossAttentionBlock(channels_hidden, channels, kernel_size, pad)

        self.topview_down0 = CrossAttentionBlock(channels_hidden, channels, kernel_size, pad)
        self.topview_down1 = CrossAttentionBlock(channels_hidden, channels, kernel_size, pad)

        self.lr = nn.ReLU()

    def forward(self, x0, x1, x2):
        # x0 is top view, x1 and x2 are side views
        out0 = self.conv_scale0_0(x0)
        out1 = self.conv_scale1_0(x1)
        out2 = self.conv_scale2_0(x2)

        y0 = self.lr(out0)
        y1 = self.lr(out1)
        y2 = self.lr(out2)

        out0 = self.conv_scale0_1(y0)
        out1 = self.conv_scale1_1(y1)
        out2 = self.conv_scale2_1(y2)

        top1_up = self.topview_up0(y1, y0)
        top2_up = self.topview_up1(y2, y0)

        top1_down = self.topview_down0(y0, y1)
        top2_down = self.topview_down1(y0, y2)

        y21 = self.cross_conv21(y1, y2)
        y12 = self.cross_conv12(y2, y1)

        out0 = out0 + 0.5 * (top1_down + top2_down)
        out1 = out1 + 0.5 * (top1_up + y21)
        out2 = out2 + 0.5 * (top2_up + y12)

        if self.use_gamma:
            out0 = out0 * self.gamma0
            out1 = out1 * self.gamma1
            out2 = out2 * self.gamma2
        return out0, out1, out2

    
# connections just as in CS-Flow, but neighboring views are connected. all side-views connect to the/ neighbor the top view 
class NeighboringSelfAttention(nn.Module):
    '''ResNet transformation, not itself reversible, just used below'''

    def __init__(self, in_channels, channels, channels_hidden=512,
                 stride=None, kernel_size=3, last_kernel_size=1, leaky_slope=0.1,
                 batch_norm=False, block_no=0, use_gamma=True):

        super(NeighboringSelfAttention, self).__init__()

        if stride:
            warnings.warn("Stride doesn't do anything, the argument should be "
                          "removed", DeprecationWarning)

        if not channels_hidden:
            channels_hidden = channels

        pad = kernel_size // 2
        self.leaky_slope = leaky_slope
        self.use_gamma = use_gamma
        pad_mode = "replicate" #  'zeros'

        self.gamma0 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.conv_scale0_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale1_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale2_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)

        self.conv_scale0_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale1_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad * 1,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale2_1 = nn.Conv2d(channels_hidden * 1, channels,  #
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)

        self.cross_conv12 = SelfAttentionBlock(channels_hidden, channels, kernel_size, pad, n_heads=1)
        self.cross_conv21 = SelfAttentionBlock(channels_hidden, channels, kernel_size, pad, n_heads=1)

        self.topview_up0 = SelfAttentionBlock(channels_hidden, channels, kernel_size, pad, n_heads=1)
        self.topview_up1 = SelfAttentionBlock(channels_hidden, channels, kernel_size, pad, n_heads=1)

        self.topview_down0 = SelfAttentionBlock(channels_hidden, channels, kernel_size, pad, n_heads=1)
        self.topview_down1 = SelfAttentionBlock(channels_hidden, channels, kernel_size, pad, n_heads=1)

        self.lr = nn.ReLU()

    def forward(self, x0, x1, x2):

        out0 = self.conv_scale0_0(x0)
        out1 = self.conv_scale1_0(x1)
        out2 = self.conv_scale2_0(x2)

        y0 = self.lr(out0)
        y1 = self.lr(out1)
        y2 = self.lr(out2)

        out0 = self.conv_scale0_1(y0)
        out1 = self.conv_scale1_1(y1)
        out2 = self.conv_scale2_1(y2)

        top1_up = self.topview_up0(y1, y0)
        top2_up = self.topview_up1(y2, y0)

        top1_down = self.topview_down0(y0, y1)
        top2_down = self.topview_down1(y0, y2)

        y21 = self.cross_conv21(y1, y2)
        y12 = self.cross_conv12(y2, y1)

        out0 = out0 + 0.5 * (top1_down + top2_down)
        out1 = out1 + 0.5 * (top1_up + y21)
        out2 = out2 + 0.5 * (top2_up + y12)

        if self.use_gamma:
            out0 = out0 * self.gamma0
            out1 = out1 * self.gamma1
            out2 = out2 * self.gamma2

        return out0, out1, out2


class SimpleThreeViewConvolution(nn.Module):
    '''Simple ResNet transformation for 3-view data without any cross-view connections.
       Used as a baseline for comparison with multi-view fusion methods.'''

    def __init__(self, in_channels, channels, channels_hidden=512,
                 stride=None, kernel_size=3, last_kernel_size=1, leaky_slope=0.1,
                 batch_norm=False, block_no=0, use_gamma=True):
        super(SimpleThreeViewConvolution, self).__init__()

        if stride:
            warnings.warn("Stride doesn't do anything, the argument should be "
                          "removed", DeprecationWarning)

        if not channels_hidden:
            channels_hidden = channels

        pad = kernel_size // 2
        self.leaky_slope = leaky_slope
        self.use_gamma = use_gamma
        pad_mode = 'replicate'

        self.gamma0 = nn.Parameter(torch.zeros(1))
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.conv_scale0_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale1_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)
        self.conv_scale2_0 = nn.Conv2d(in_channels, channels_hidden,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)

        self.conv_scale0_1 = nn.Conv2d(channels_hidden * 1, channels,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale1_1 = nn.Conv2d(channels_hidden * 1, channels,
                                       kernel_size=kernel_size, padding=pad * 1,
                                       bias=not batch_norm, padding_mode=pad_mode, dilation=1)
        self.conv_scale2_1 = nn.Conv2d(channels_hidden * 1, channels,
                                       kernel_size=kernel_size, padding=pad,
                                       bias=not batch_norm, padding_mode=pad_mode)

        self.lr = nn.ReLU()

    def forward(self, x0, x1, x2):
        # x0 is top view, x1 and x2 are side views
        out0 = self.conv_scale0_0(x0)
        out1 = self.conv_scale1_0(x1)
        out2 = self.conv_scale2_0(x2)

        y0 = self.lr(out0)
        y1 = self.lr(out1)
        y2 = self.lr(out2)

        out0 = self.conv_scale0_1(y0)
        out1 = self.conv_scale1_1(y1)
        out2 = self.conv_scale2_1(y2)

        out0 = out0
        out1 = out1
        out2 = out2

        if self.use_gamma:
            out0 = out0 * self.gamma0
            out1 = out1 * self.gamma1
            out2 = out2 * self.gamma2

        return out0, out1, out2




class ParallelPermute(nn.Module):
    '''permutes input vector in a random but fixed way'''

    def __init__(self, dims_in, seed):
        super(ParallelPermute, self).__init__()

        self.n_inputs = len(dims_in)

        self.in_channels = [dims_in[i][0] for i in range(self.n_inputs)]

        np.random.seed(seed)

        self.perm = []
        self.perm_inv = []
        for i in range(self.n_inputs):
            perm, perm_inv = self.get_random_perm(i)
            self.perm.append(perm)
            self.perm_inv.append(perm_inv)

    def get_random_perm(self, i):
        perm = np.random.permutation(self.in_channels[i])
        perm_inv = np.zeros_like(perm)
        for i, p in enumerate(perm):
            perm_inv[p] = i

        perm = torch.LongTensor(perm)
        perm_inv = torch.LongTensor(perm_inv)
        return perm, perm_inv

    def forward(self, x, rev=False):
        if not rev:
            return [x[i][:, self.perm[i]] for i in range(self.n_inputs)]
        else:
            return [x[i][:, self.perm_inv[i]] for i in range(self.n_inputs)]

    def jacobian(self, x, rev=False):
        return [0.] * self.n_inputs

    def output_dims(self, input_dims):
        return input_dims



class MVFC_block(nn.Module):
    def __init__(self, dims_in, F_class, F_args={},
                 clamp=5., use_noise=False):
        super(MVFC_block, self).__init__()

        channels = dims_in[0][0]
        self.ndims = len(dims_in[0])
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2

        self.clamp = clamp
        self.max_s = exp(clamp)
        self.min_s = exp(-clamp)

        self.use_noise = use_noise
        self.cond_dim = 1 if self.use_noise else 0

        self.s1 = F_class(self.split_len1 + self.cond_dim, self.split_len2 * 2, **F_args)
        self.s2 = F_class(self.split_len2 + self.cond_dim, self.split_len1 * 2, **F_args)

    def e(self, s):
        if self.clamp > 0:
            return torch.exp(self.log_e(s))
        else:
            return torch.exp(s)

    def log_e(self, s):
        if self.clamp > 0:
            return self.clamp * 0.636 * torch.atan(s / self.clamp)
        else:
            return s

    def forward(self, x, rev=False):
        if self.use_noise:
            def c(a,b):
                return torch.cat((a,b), dim=1)
            cond = x[3]

        x01, x02 = (x[0].narrow(1, 0, self.split_len1),
                    x[0].narrow(1, self.split_len1, self.split_len2))
        x11, x12 = (x[1].narrow(1, 0, self.split_len1),
                    x[1].narrow(1, self.split_len1, self.split_len2))
        x21, x22 = (x[2].narrow(1, 0, self.split_len1),
                    x[2].narrow(1, self.split_len1, self.split_len2))

        if not rev:
            if self.use_noise:
                r02, r12, r22 = self.s2(c(x02, cond), c(x12, cond), c(x22, cond))
            else:
                r02, r12, r22 = self.s2(x02, x12, x22)

            s02, t02 = r02[:, :self.split_len1], r02[:, self.split_len1:]
            s12, t12 = r12[:, :self.split_len1], r12[:, self.split_len1:]
            s22, t22 = r22[:, :self.split_len1], r22[:, self.split_len1:]

            y01 = self.e(s02) * x01 + t02
            y11 = self.e(s12) * x11 + t12
            y21 = self.e(s22) * x21 + t22

            if self.use_noise:
                r01, r11, r21 = self.s1(c(y01, cond), c(y11, cond), c(y21, cond))
            else:
                r01, r11, r21 = self.s1(y01, y11, y21)

            s01, t01 = r01[:, :self.split_len2], r01[:, self.split_len2:]
            s11, t11 = r11[:, :self.split_len2], r11[:, self.split_len2:]
            s21, t21 = r21[:, :self.split_len2], r21[:, self.split_len2:]

            y02 = self.e(s01) * x02 + t01
            y12 = self.e(s11) * x12 + t11
            y22 = self.e(s21) * x22 + t21

        else:
            raise NotImplementedError("Reverse is not needed for inference in AD; therefore it's not implemented!")

        y0 = torch.cat((y01, y02), 1)
        y1 = torch.cat((y11, y12), 1)
        y2 = torch.cat((y21, y22), 1)

        y0 = torch.clamp(y0, -1e6, 1e6)
        y1 = torch.clamp(y1, -1e6, 1e6)
        y2 = torch.clamp(y2, -1e6, 1e6)

        jac0 = torch.sum(self.log_e(s01), dim=(1,)) + torch.sum(self.log_e(s02), dim=(1,))
        jac1 = torch.sum(self.log_e(s11), dim=(1,)) + torch.sum(self.log_e(s12), dim=(1,))
        jac2 = torch.sum(self.log_e(s21), dim=(1,)) + torch.sum(self.log_e(s22), dim=(1,))

        self.last_jac = [jac0, jac1, jac2]

        return [y0, y1, y2]

    def jacobian(self, x, rev=False):
        return self.last_jac

    def output_dims(self, input_dims):
        return input_dims



