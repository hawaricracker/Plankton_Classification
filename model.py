import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, use_act=True):
        super().__init__()
        self.act = nn.SiLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.use_act = use_act

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        if self.use_act:
            out = self.act(out)
        return out

class DWConv(nn.Module):
    def __init__(self, in_channels, stride=2):
        super().__init__()
        self.network = ConvBNAct(in_channels, in_channels, groups=in_channels, stride=stride)

    def forward(self, x):
        return self.network(x)

class HGStem(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.stem1 = ConvBNAct(in_channels, mid_channels, kernel_size=3, stride=2)
        self.stem2 = ConvBNAct(mid_channels, mid_channels, kernel_size=2, stride=1, padding=0)
        self.stem3 = ConvBNAct(mid_channels * 2, mid_channels, kernel_size=3, stride=2)
        self.stem4 = ConvBNAct(mid_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, ceil_mode=True)

    def _pad_same(self, x, kernel_size=2):
        x = F.pad(x, (0, 1, 0, 1))
        return x

    def forward(self, x):
        x = self.stem1(x)

        x2 = self._pad_same(x)
        x2 = self.stem2(x2)

        x1 = self._pad_same(x)
        x1 = self.pool(x1)

        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x

class GhostBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ratio=2):
        super().__init__()
        init_ch = out_channels // ratio
        new_ch = out_channels - init_ch

        self.primary = ConvBNAct(in_channels, init_ch, 1, stride=1, padding=0, use_act=False)
        self.cheap_dw = ConvBNAct(init_ch, init_ch, kernel_size, stride=1, padding=kernel_size//2, groups=init_ch)
        self.cheap_pw = ConvBNAct(init_ch, new_ch, 1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.primary(x)
        x2 = self.cheap_dw(x1)
        x2 = self.cheap_pw(x2)
        return torch.cat([x1, x2], dim=1)

class ESE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        weight = self.gap(x)
        weight = self.sigmoid(self.fc(weight))
        return x * weight

class H2GBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, layer_num=6):
        super().__init__()
        out_channels = in_channels
        self.layers = nn.ModuleList()
        for i in range(layer_num):
            self.layers.append(GhostBlock(in_channels if i == 0 else mid_channels, mid_channels, 3))

        total_ch = in_channels + layer_num * mid_channels
        self.conv1 = ConvBNAct(total_ch, out_channels, kernel_size=1, padding=0)
        self.ese = ESE(out_channels)

    def forward(self, x):
        identity = x
        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)
        x = torch.cat(output, dim=1)
        x = self.conv1(x)
        x = self.ese(x)
        x += identity
        return x

class Bottleneck(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.network = nn.Sequential(ConvBNAct(in_channels, in_channels), ConvBNAct(in_channels, in_channels))

    def forward(self, x):
        identity = x
        x = self.network(x)
        x += identity
        return x

class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, num_layer=3):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels)
        self.bottlenecks = nn.ModuleList(Bottleneck(out_channels // 2) for _ in range(num_layer))
        self.conv2 = ConvBNAct(int(0.5 * (num_layer + 2) * out_channels), out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        chunks = list(x.chunk(2, dim=1))

        for bottleneck in self.bottlenecks:
            chunks.append(bottleneck(chunks[-1]))

        x = torch.cat(chunks, dim=1)
        x = self.conv2(x)
        return x

class ADown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels // 2, out_channels // 2, stride=2)
        self.conv2 = ConvBNAct(in_channels // 2, out_channels // 2, kernel_size=1, padding=0)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.pad(x, (0, 1, 0, 1))
        x = self.avgpool(x)
        chunks = list(x.chunk(2, dim=1))
        x1 = self.conv1(chunks[0])
        x2 = self.maxpool(chunks[1])
        x2 = self.conv2(x2)
        x = torch.cat([x1, x2], dim=1)
        return x

class ConvGN(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups, kernel_size=3, stride=1, padding=1, groups=1, use_act=True):
        super().__init__()
        self.act = nn.SiLU()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups)
        self.gn = nn.GroupNorm(num_groups, out_channels)
        self.use_act = use_act

    def forward(self, x):
        out = self.conv(x)
        out = self.gn(out)
        if self.use_act:
            out = self.act(out)
        return out

class LSCD_Clsf(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.network = nn.Sequential(
            ConvGN(in_channels, out_channels, out_channels // 2),
            ConvGN(out_channels, out_channels, out_channels // 4),
            ConvGN(out_channels, out_channels, 1))
        self.linear = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, x):
        x = self.network(x)
        x = self.linear(x)
        return x

class Plank_li_net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone1 = nn.Sequential(HGStem(1, 16, 32), H2GBlock(32, 32), DWConv(32), H2GBlock(32, 32))
        self.backbone2 = nn.Sequential(DWConv(32), H2GBlock(32, 32), H2GBlock(32, 32))
        self.backbone3 = nn.Sequential(DWConv(32), H2GBlock(32, 32))
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.c2f_b3_b2 = C2f(64, 64)
        self.c2f_b2_b1 = C2f(96, 96)
        self.adown1 = ADown(96, 64)
        self.adown2 = ADown(128, 96)
        self.c2f0 = C2f(128, 128)
        self.c2f1 = C2f(128, 128)
        self.lscd1 = LSCD_Clsf(96, 96)
        self.lscd2 = LSCD_Clsf(128, 128)
        self.lscd3 = LSCD_Clsf(128, 128)
        self.linear = nn.Linear(352, num_classes)

    def forward(self, x):
        b1 = self.backbone1(x)
        b2 = self.backbone2(b1)
        b3 = self.backbone3(b2)
        b3_up = self.upsample(b3)
        b3_b2_cat = torch.cat([b2, b3_up], dim=1)
        b3_b2_cat = self.c2f_b3_b2(b3_b2_cat)
        b2_up = self.upsample(b3_b2_cat)
        b2_b1_cat = torch.cat([b1, b2_up], dim=1)
        b2_b1_cat = self.c2f_b2_b1(b2_b1_cat)
        adown_b2_b1 = self.adown1(b2_b1_cat)
        adown_b3_b2_cat = torch.cat([adown_b2_b1, b3_b2_cat], dim=1)
        adown_b3_b2_cat = self.c2f0(adown_b3_b2_cat)
        adown_last = self.adown2(adown_b3_b2_cat)
        adown_last_b3_cat = torch.cat([adown_last, b3], dim=1)
        adown_last_b3_cat = self.c2f1(adown_last_b3_cat)
        final_cat = torch.cat([self.lscd1(b2_b1_cat), self.lscd2(adown_b3_b2_cat), self.lscd3(adown_last_b3_cat)], dim=1)
        return self.linear(final_cat)

class SiReconBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1), nn.BatchNorm2d(out_ch)
        )
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.net(x) + self.proj(x))

class Plank_SiRecon(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            SiReconBlock(1, 32), nn.MaxPool2d(2),
            SiReconBlock(32, 64), nn.MaxPool2d(2),
            SiReconBlock(64, 128), nn.MaxPool2d(2),
            SiReconBlock(128, 256),
        )
        self.linear = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.linear(self.network(x))

class SiDSCConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        self.relu = nn.ReLU()
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels), nn.BatchNorm2d(in_channels), nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels), nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.network(x)
        out = out + self.proj(x)
        return self.relu(out)

class Plank_SiDSC(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.linear = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, num_classes))
        self.network = nn.Sequential(
            SiDSCConv(1, 64), nn.MaxPool2d(2),
            SiDSCConv(64, 128), nn.MaxPool2d(2),
            SiDSCConv(128, 256), nn.MaxPool2d(2),
            SiDSCConv(256, 512),
        )

    def forward(self, x):
        out = self.network(x)
        return self.linear(out)

import torch
import torch.nn as nn
import math

def window_partition(x, window_size):
    B,H,W,C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    x = x.permute(0,1,3,2,4,5).contiguous().view(-1, window_size, window_size, C)
    return x

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0]/(H*W/window_size/window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class PatchEmbed(nn.Module):
    def __init__(self, input_ch=3, dim=10, kernel_size=4, padding=0):
        super().__init__()
        self.embedding = nn.Conv2d(in_channels=input_ch, out_channels=dim, kernel_size=kernel_size, stride=kernel_size, padding=padding)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        x = self.embedding(x)
        x_spatial = x
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, x_spatial
    
class WindowAttention(nn.Module):
    def __init__(self, window_size=7, dim=10, num_heads=2):
        super().__init__()
        assert dim % num_heads == 0
        self.window_size = window_size
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)

        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)

        relative_coords = (coords_flatten[:, :, None] - coords_flatten[:, None, :])

        relative_coords = (relative_coords.permute(1, 2, 0).contiguous())

        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1

        relative_coords[:, :, 0] *= (2 * window_size - 1)

        relative_position_index = relative_coords.sum(-1)

        self.register_buffer("relative_position_index", relative_position_index)

        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

    def forward(self, x, mask=None):

        B_, N, C = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(B_, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = q @ k.transpose(-2, -1)
        attn = attn / (self.head_dim ** 0.5)

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(N, N, self.num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = torch.softmax(attn, dim=-1)
        x = attn @ v
        x = x.transpose(1, 2)
        x = x.reshape(B_, N, C)
        x = self.proj(x)

        return x

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.fc1 = nn.Linear(dim, dim * 4)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim=10):
        super().__init__()

        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.input_resolution = input_resolution

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        assert L == H * W
        assert H % 2 == 0 and W % 2 == 0

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        x = torch.cat([x0, x1, x2, x3], dim=-1)

        x = x.view(B, -1, 4 * C)

        x = self.norm(x)
        x = self.reduction(x)

        return x

class SwinTransformerBlock(nn.Module):
    #Patch Embed terlebih dahulu
    def __init__(self, input_resolution, window_size=7, dim=10, num_heads=2, shift_size=0):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim=dim)
        self.shift_size = shift_size
        self.input_resolution = input_resolution

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.attn = WindowAttention(window_size=self.window_size, dim=dim, num_heads=num_heads)

        if self.shift_size > 0:
            H, W = input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)
            )

            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None)
            )

            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

            attn_mask = (mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2))
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None
        
        if attn_mask is not None:
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.attn_mask = None

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape

        assert L == H * W

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        attn_mask = self.attn_mask

        
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        x_windows = window_partition(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        x = x.view(B, H * W, C)
        x = shortcut + x

        shortcut = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x

        return x

class Basic_layer(nn.Module):
    def __init__(self, input_resolution, dim, num_heads, stages, window_size=7):
        super().__init__()
        self.blocks = nn.ModuleList([SwinTransformerBlock(input_resolution=input_resolution, dim=dim, num_heads=num_heads, 
                                                          window_size=window_size, 
                                                          shift_size=0 if i % 2 == 0 else window_size // 2)
                                     for i in range(stages)])
    
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x
    
class DPCB(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.path1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(7,3), stride=stride, padding=(3,1))
        self.path2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,7), stride=stride, padding=(1,3))

    def forward(self, x):
        p1 = self.path1(x)
        p2 = self.path2(x)
        x_spatial = p1 + p2
        x = x_spatial.flatten(2)
        x = x.transpose(1,2)
        return x, x_spatial

class EMA(nn.Module):
    def __init__(self, channels, groups=8):
        super().__init__()
        self.groups = groups
        assert channels % groups == 0
        self.C_per_G = channels // groups

        self.concat_conv = nn.Conv2d(self.C_per_G * 2, self.C_per_G, kernel_size=1)
        self.sigmoid_w = nn.Sigmoid()
        self.sigmoid_h = nn.Sigmoid()
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=channels)
        self.conv3x3 = nn.Conv2d(self.C_per_G, self.C_per_G, kernel_size=3, padding=1)
    
    def forward(self, x):
        B, L, C = x.shape
        x = x.transpose(1,2).view(B, C, int(L**0.5), int(L**0.5))
        B, C, H, W = x.shape
        G = self.groups
        CG = C // G
        x_grouped = x.view(B * G, CG, H, W)

        #Left
        x_pool_h = x_grouped.mean(dim=3, keepdim=True)
        x_pool_w = x_grouped.mean(dim=2, keepdim=True)
        x_pool_h_exp = x_pool_h.expand(-1, -1, H, W)
        x_pool_w_exp = x_pool_w.expand(-1, -1, H, W)
        concat = torch.cat([x_pool_h_exp, x_pool_w_exp], dim=1)

        fused = self.concat_conv(concat)
        attn_w = self.sigmoid_w(fused.mean(dim=2, keepdim=True))
        attn_h = self.sigmoid_h(fused.mean(dim=3, keepdim=True))

        x_spatial = x_grouped * attn_h * attn_w
        x_spatial = x_spatial.view(B, C, H, W)

        x_norm = self.group_norm(x_spatial)
        x_norm_g = x_norm.view(B * G, CG, H, W)
        x_gap_l = x_norm_g.mean(dim=[2, 3], keepdim=True)
        x_gap_flat_l = x_gap_l.view(B * G, 1, CG)
        x_softmax_l = torch.softmax(x_gap_flat_l, dim=-1)

        #Right
        x_r = self.conv3x3(x_grouped)
        x_gap_r = x_r.mean(dim=[2, 3], keepdim=True)
        x_gap_flat_r = x_gap_r.view(B * G, 1, CG)
        x_softmax_r = torch.softmax(x_gap_flat_r, dim=-1)

        #Matmul
        x_flat_norm = x_norm_g.view(B * G, CG, H * W)
        x_flat_r = x_r.view(B * G, CG, -1)
        matmul_l = torch.matmul(x_softmax_l, x_flat_r)
        matmul_r = torch.matmul(x_softmax_r, x_flat_norm)

        x_sum = matmul_l + matmul_r
        x_sum = x_sum.view(B * G, 1, H, W)
        x_sum = torch.sigmoid(x_sum)
        x_out = x_grouped * x_sum
        x_out = x_out.view(B, C, H, W)
        x_out = x_out.view(B, C, L).transpose(1, 2)
        return x_out

class RFEM(nn.Module):
    def __init__(self, input_resolution, dim, num_heads, stages, window_size=7):
        super().__init__()
        self.groups = math.ceil(stages / 6.0)
        last_group = stages % 6

        swt_groups = []
        for i in range(self.groups):
            block = nn.ModuleList([
                SwinTransformerBlock(
                    input_resolution=input_resolution, dim=dim, num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if j % 2 == 0 else window_size // 2)
                for j in range(6)
            ])
            swt_groups.append(block)

        if last_group != 0:
            swt_groups[-1] = nn.ModuleList([
                SwinTransformerBlock(
                    input_resolution=input_resolution, dim=dim, num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if j % 2 == 0 else window_size // 2)
                for j in range(last_group)
            ])

        self.swt_groups = nn.ModuleList(swt_groups)
        self.ema_modules = nn.ModuleList([EMA(channels=dim) for _ in range(self.groups)])

    def forward(self, x):
        for i, (group, ema) in enumerate(zip(self.swt_groups, self.ema_modules)):
            residual = x
            for block in group:
                x = block(x)
            x = x + ema(residual)
        return x

class HySwinFormer(nn.Module):
    def __init__(self, input_resolution, num_classes=1000):
        super().__init__()
        H, W = input_resolution
        self.Pembed = PatchEmbed(input_ch=1, dim=96)
        self.stage1 = Basic_layer(input_resolution=(H // 4, W // 4), dim=96, num_heads=3, stages=2)
        self.stage2 = Basic_layer(input_resolution=(H // 8, W // 8), dim=192, num_heads=6, stages=2)
        self.stage3 = RFEM(input_resolution=(H // 16, W // 16), dim=384, num_heads=12, stages=6)
        self.stage4 = Basic_layer(input_resolution=(H // 32, W // 32), dim=768, num_heads=24, stages=2)
        self.dpcb1 = DPCB(96, 192)
        self.dpcb2 = DPCB(192, 384)
        self.dpcb3 = DPCB(384, 768)
        self.dpcb4 = DPCB(768, 768, stride=1)
        self.pmerging1 = PatchMerging(input_resolution=(H // 4, W // 4), dim=96)
        self.pmerging2 = PatchMerging(input_resolution=(H // 8, W // 8), dim=192)
        self.pmerging3 = PatchMerging(input_resolution=(H // 16, W // 16), dim=384)
        self.norm = nn.LayerNorm(768)
        self.avgpool_swin = nn.AdaptiveAvgPool1d(1)
        self.avgpool_cnn = nn.AdaptiveAvgPool2d(1)
        self.head_swin = nn.Linear(768, num_classes)
        self.head_cnn = nn.Linear(768, num_classes)
        self.alpha1 = nn.Parameter(torch.tensor(0.5))
        self.alpha2 = nn.Parameter(torch.tensor(0.5))
        self.alpha3 = nn.Parameter(torch.tensor(0.5))
        self.alpha4 = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        split = self.Pembed(x)
        deb1 = self.dpcb1(split[1])
        stage1 = self.stage1(split[0])
        stage1 = self.pmerging1(stage1)
        a1 = torch.sigmoid(self.alpha1)
        stage1 = a1 * deb1[0] + (1 - a1) * stage1

        stage2 = self.stage2(stage1)
        deb2 = self.dpcb2(deb1[1])
        stage2 = self.pmerging2(stage2)
        a2 = torch.sigmoid(self.alpha2)
        stage2 = a2 * deb2[0] + (1 - a2) * stage2

        stage3 = self.stage3(stage2)
        deb3 = self.dpcb3(deb2[1])
        stage3 = self.pmerging3(stage3)
        a3 = torch.sigmoid(self.alpha3)
        stage3 = a3 * deb3[0] + (1 - a3) * stage3

        stage4 = self.stage4(stage3)
        deb4 = self.dpcb4(deb3[1])
        a4 = torch.sigmoid(self.alpha4)
        stage4 = a4 * deb4[0] + (1 - a4) * stage4

        # Swin branch
        x_swin = self.norm(stage4)
        x_swin = self.avgpool_swin(x_swin.transpose(1, 2))
        x_swin = torch.flatten(x_swin, 1)
        x_swin = self.head_swin(x_swin)

        # CNN branch
        x_cnn = self.avgpool_cnn(deb4[1])
        x_cnn = torch.flatten(x_cnn, 1)
        x_cnn = self.head_cnn(x_cnn)

        return x_swin + x_cnn

def create_model(model_name, num_classes, device, input_size=None):
    """Buat model berdasarkan nama, pindahkan ke device."""
    if model_name == 'linet':
        model = Plank_li_net(num_classes).to(device)
    elif model_name == 'Si-Recon':
        model = Plank_SiRecon(num_classes).to(device)
    elif model_name == 'Si-DSC':
        model = Plank_SiDSC(num_classes).to(device)
    elif model_name == 'HySwinFormer':
        assert input_size is not None, "HySwinFormer requires --size"
        model = HySwinFormer(input_resolution=(input_size, input_size), num_classes=num_classes).to(device)
    elif model_name == 'yolo':
        obj = torch.load("yolo26m-cls.pt", weights_only=False)
        model = obj['model']
        old_conv = model.model[0].conv
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        model.model[0].conv = new_conv
        old_linear = model.model[-1].linear
        new_linear = nn.Linear(in_features=old_linear.in_features, out_features=num_classes, bias=old_linear.bias is not None)
        model.model[-1].linear = new_linear
        model = model.float()
        for name, layer in model.named_modules():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
                print(f'Reset:{name}')
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown model: {model_name}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,}  |  Trainable: {trainable:,}")
    return model
