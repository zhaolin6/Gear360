import torch
from torch import nn
import torch.nn.functional as F
import math

from .cv_couplings import (
    NaiveCrossConvolutions,
    NeighboringCrossConvolutions,
    NeighboringCrossAttention,
    NeighboringSelfAttention,
    MVFC_block,
    ParallelPermute,
    CBST,
    SimpleThreeViewConvolution,
)
from .freia_funcs import (
    InputNode,
    Node,
    OutputNode,
    ReversibleGraphNet
)


def flat(tensor):
    return tensor.reshape(tensor.shape[0], -1)


def concat_maps(maps):
    flat_maps = list()
    for m in maps:
        flat_maps.append(flat(m))
    return torch.cat(flat_maps, dim=1)[..., None]


def cat_maps(z):
    return torch.cat([z[i].reshape(z[i].shape[0], -1) for i in range(len(z))], dim=1)

def get_cs_flow_model(config):
    input_dim = config["n_feat"]
    map_len = config["map_len"]
    nodes = list()
    if config["use_noise"]:
        nodes.append(InputNode(1, map_len, map_len, name="input0"))

    nodes.append(InputNode(input_dim, map_len, map_len, name='input1'))
    nodes.append(InputNode(input_dim, map_len, map_len, name='input2'))
    nodes.append(InputNode(input_dim, map_len, map_len, name='input3'))

    cross_convolution = {
        "cs_naive": NaiveCrossConvolutions,
        "cs_neigh": NeighboringCrossConvolutions,
        "cs_att_cross": NeighboringCrossAttention,
        "cs_att_self": NeighboringSelfAttention,
        "cs_CBST": CBST,
        "cs_STVC": SimpleThreeViewConvolution,
    }[config["arch"]]

    for k in range(config["n_coupling_blocks"]):
        if k == 0:
            node_to_permute = [nodes[-3].out0, nodes[-2].out0, nodes[-1].out0]
        else:
            node_to_permute = [nodes[-1].out0, nodes[-1].out1, nodes[-1].out2]
        nodes.append(Node(node_to_permute, ParallelPermute,
                          {'seed': k}, name=F'permute_{k}'))
        input_list = [nodes[-1].out0, nodes[-1].out1, nodes[-1].out2]
        if config["use_noise"]:
            input_list.append(nodes[0].out0)
        nodes.append(Node(input_list,
                          MVFC_block,
                          {'clamp': config["clamp"],
                           'F_class': cross_convolution,
                           "use_noise": config["use_noise"],
                           'F_args': {
                               'channels_hidden': config["channels_hidden_teacher"],
                               'kernel_size': config["kernel_sizes"][k],
                               'block_no': k}},
                          name=F'fc1_{k}'))

    nodes.append(OutputNode([nodes[-1].out0], name='output_end0'))
    nodes.append(OutputNode([nodes[-2].out1], name='output_end1'))
    nodes.append(OutputNode([nodes[-3].out2], name='output_end2'))

    nf = ReversibleGraphNet(nodes, n_jac=3)
    return nf




class FESA(nn.Module):
    def __init__(self, lamda=1e-4, eps=1e-6, topk_ratio=0.1, alpha=0.4, temperature=1):
        super().__init__()
        self.lamda = lamda
        self.eps = eps
        self.topk_ratio = topk_ratio
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, x):
        B, C, H, W = x.shape
        S = H * W
        n = max(S - 1, 1)

        mean = x.mean(dim=(2, 3), keepdim=True)
        var  = ((x - mean) ** 2).sum(dim=(2, 3), keepdim=True) / n
        e_t  = (x - mean) ** 2 / (4.0 * (var + self.lamda)) + 0.5
        w_s  = torch.sigmoid(e_t)

        # --- channel score: TopK / Mean (peak-to-background ratio) ---
        k = max(1, int(S * self.topk_ratio))
        e_flat = e_t.view(B, C, S)
        topk_mean = torch.topk(e_flat, k=k, dim=-1, largest=True, sorted=False).values.mean(dim=-1)  # [B,C]
        mean_all  = e_flat.mean(dim=-1) + self.eps                                                   # [B,C]
        s = topk_mean / mean_all                                                                      # [B,C]

        # sample-wise normalize (optional but usually helps)
        s = s - s.mean(dim=1, keepdim=True)
        s = s / (s.std(dim=1, keepdim=True) + self.eps)
        w_c = torch.sigmoid(s / self.temperature).view(B, C, 1, 1)

        gate = 1.0 + self.alpha * (w_c - 0.5)
        return x * w_s * gate

class EMA(nn.Module):
    def __init__(self, channels, factor=8):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0, "Channels must be divisible by groups"

        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)

        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()

        group_x = x.reshape(b * self.groups, -1, h, w)

        x_h = self.pool_h(group_x)  # [B*G, C/G, H, 1]
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)  # [B*G, C/G, W, 1]

        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)

        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())

        x2 = self.conv3x3(group_x)

        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)

        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)

        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)

        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, scaling=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // scaling, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // scaling, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv1(x_cat)
        return self.sigmoid(x_out)


class CBAM_Attention(nn.Module):
    def __init__(self, channel, scaling=16, kernel_size=7):
        super(CBAM_Attention, self).__init__()
        self.channelattention = ChannelAttention(channel, scaling=scaling)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x

class PAM(nn.Module):

    def __init__(self, in_chans: int):
        super(PAM, self).__init__()
        self.in_chans = in_chans
        self.q = nn.Conv2d(in_chans, in_chans // 8, kernel_size=1)
        self.k = nn.Conv2d(in_chans, in_chans // 8, kernel_size=1)
        self.v = nn.Conv2d(in_chans, in_chans, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.size()
        # (B, HW, C/8)
        q = self.q(x).view(b, -1, h * w).permute(0, 2, 1)
        # (B, C/8, HW)
        k = self.k(x).view(b, -1, h * w)
        # (B, C, HW)
        v = self.v(x).view(b, -1, h * w)

        # (B, HW, HW)
        attn = self.softmax(torch.bmm(q, k))
        # (B, C, HW) = (B, C, HW) @ (B, HW, HW)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(b, c, h, w)

        return self.gamma * out + x

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if not config["pre_extracted"]:
            raise NotImplementedError("Please pre-extract using the preprocess.py!")
        self.use_noise = config["use_noise"]
        self.net = get_cs_flow_model(config)
        self.config = config
        self.fesa = FESA(lamda=1e-4, topk_ratio=0.1, alpha=0.4)

    def loss(self, z, jac, per_sample=False, per_pixel=False, mask=None, means=0, n_views=3):
        B = z[0].shape[0]
        z = torch.cat(z, dim=0)
        idx = torch.arange(z.shape[0])
        result = (idx % n_views) * B + (idx // n_views)

        pixel_scores = F.relu(0.5 * torch.sum((mask.unsqueeze(1) * z[result, ...] - means) ** 2, dim=1)
                              - mask * jac[result, ...]) / z.shape[1]

        if per_pixel:
            return pixel_scores
        elif per_sample:
            return pixel_scores.mean(dim=(-1, -2))
        return pixel_scores.mean()


    def forward(self, x):
        if not self.config["pre_extracted"]:
            with torch.no_grad():
                f = self.feature_extractor(x)
        else:
            f = x

        if isinstance(f, list):
            f = [self.fesa(view_feat) for view_feat in f]
        else:
            f = self.fesa(f)

        if self.use_noise:
            b_size = x[0].shape[0]
            if self.use_noise == 1:
                c = (torch.tensor([0.15 for _ in range(b_size)]) if self.training
                     else torch.zeros(b_size))[..., None, None, None]
                noise = torch.randn((b_size, 1, self.config["map_len"],
                                     self.config["map_len"])) * c
            elif self.use_noise == 2:
                c = (torch.rand(b_size) if self.training else torch.zeros(b_size))[..., None, None, None]
                noise = torch.randn((b_size, 1, self.config["map_len"],
                                     self.config["map_len"])) * (c * c)
            elif self.use_noise == 3:
                noise = torch.rand((b_size, 1, self.config["map_len"],
                                    self.config["map_len"]))
                if not self.training:
                    noise *= 0
            noise = noise.to(self.config["device"])

            if isinstance(f, list):
                f = [x_entry + noise for x_entry in f]
                f = [noise * 20, *f]
            else:
                f = f + noise
                f = [noise * 20, f]
        inp = f
        z = self.net(inp)
        jac = torch.cat(self.net.jacobian(run_forward=False), dim=0)
        return z, jac

