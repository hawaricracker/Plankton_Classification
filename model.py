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

class ResBlock(nn.Module):
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

class Plank_Resnet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.network = nn.Sequential(
            ResBlock(1, 32), nn.MaxPool2d(2),
            ResBlock(32, 64), nn.MaxPool2d(2),
            ResBlock(64, 128), nn.MaxPool2d(2),
            ResBlock(128, 256),
        )
        self.linear = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.linear(self.network(x))

class ResDepthConv(nn.Module):
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

class Plank_Resdepth(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.linear = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, num_classes))
        self.network = nn.Sequential(
            ResDepthConv(1, 64), nn.MaxPool2d(2),
            ResDepthConv(64, 128), nn.MaxPool2d(2),
            ResDepthConv(128, 256), nn.MaxPool2d(2),
            ResDepthConv(256, 512),
        )

    def forward(self, x):
        out = self.network(x)
        return self.linear(out)

def create_model(model_name, num_classes, device):
    """Buat model berdasarkan nama, pindahkan ke device."""
    if model_name == 'linet':
        model = Plank_li_net(num_classes).to(device)
    elif model_name == 'resnet':
        model = Plank_Resnet(num_classes).to(device)
    elif model_name == 'resdepth':
        model = Plank_Resdepth(num_classes).to(device)
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
