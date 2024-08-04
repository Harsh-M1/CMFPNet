import torch
import torch.nn as nn
from einops import rearrange
import torch
from torch.nn import functional as F


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_2d(inp, oup, kernel_size=3, stride=1, padding=0, groups=1, bias=False, norm=True, act=True):
    conv = nn.Sequential()
    conv.add_module('conv', nn.Conv2d(inp, oup, kernel_size,
                    stride, padding, bias=bias, groups=groups))
    if norm:
        conv.add_module('BatchNorm2d', nn.BatchNorm2d(oup))
    if act:
        conv.add_module('Activation', nn.SiLU())
    return conv


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        # print('pool!', x.shape)
        return self.pool(x) - x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PoolFormer(nn.Module):
    def __init__(self, dim, depth, mlp_ratio=4, pool_size=3,
                 act_layer=nn.GELU,
                 drop=0.):

        super().__init__()

        self.norm1 = nn.GroupNorm(1, dim)
        self.norm2 = nn.GroupNorm(1, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.drop_path = nn.Identity()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Pooling(pool_size=pool_size),
                Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                    act_layer=act_layer, drop=drop)
            ]))

    def forward(self, x):
        for pool, mlp in self.layers:
            x = x + pool(self.norm1(x))
            x = x + mlp(self.norm2(x))
        return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=2):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride,
                          1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# Local–Global Perception Block (LGPB)


class LGPB(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_ratio, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size
        # local representation
        self.local_rep = nn.Sequential()
        self.local_rep.add_module('conv_3x3', conv_2d(
            channel, channel, kernel_size=3, stride=1, padding=1, groups=channel))
        self.local_rep.add_module('conv_1x1', conv_2d(
            channel, dim, kernel_size=1, stride=1, norm=False, act=False))
        # global rep
        self.global_proj = conv_2d(
            channel, dim, kernel_size=1, stride=1, norm=False, act=False)
        self.poolformer = PoolFormer(dim, depth, mlp_ratio)

        self.conv_proj = conv_1x1_bn(dim, channel)
        self.conv_out = conv_1x1_bn(dim+channel, channel)
        # -------------------------------------------------
        self.patch_area = self.pw * self.ph

    def forward(self, x):
        res = x.clone()

        # Local representations
        fm_conv = self.local_rep(x)
        # Global representations

        fg = self.global_proj(x)
        _, _, h, w = fg.shape
        # print(fm_conv.shape)
        fg = rearrange(
            fg, 'b c (h ph) (w pw) -> b c (ph pw) (h w)', ph=self.ph, pw=self.pw)
        # print(x.shape)
        fg = self.poolformer(fg)
        fg = rearrange(fg, 'b c (ph pw) (h w) -> b c (h ph) (w pw)',
                       h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)
        fm = self.conv_proj(fg)

        # Fusion
        x = self.conv_out(torch.cat((fm, fm_conv), dim=1))
        x = x + res
        return x


class Encoder(nn.Module):
    def __init__(self, image_size, dims, channels, inchannels, expansion=2, kernel_size=3, patch_size=(2, 2)):
        super().__init__()
        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        L = [2, 4, 4]

        self.conv1 = conv_nxn_bn(inchannels, channels[0], stride=2)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(InvertedResidual(
            channels[0], channels[1], 1, expansion))
        self.mv2.append(InvertedResidual(
            channels[1], channels[2], 2, expansion))
        self.mv2.append(InvertedResidual(
            channels[2], channels[3], 1, expansion))
        self.mv2.append(
            InvertedResidual(channels[2], channels[3], 1, expansion))   # Repeat
        self.mv2.append(InvertedResidual(
            channels[3], channels[4], 2, expansion))
        self.mv2.append(InvertedResidual(
            channels[5], channels[6], 2, expansion))
        self.mv2.append(InvertedResidual(
            channels[7], channels[8], 2, expansion))

        self.poolt = nn.ModuleList([])
        self.poolt.append(InvertedResidual(
            channels[4], channels[5], 1, expansion))
        self.poolt.append(InvertedResidual(
            channels[6], channels[7], 1, expansion))
        self.poolt.append(LGPB(
            dims[2], L[2], channels[9], kernel_size, patch_size, mlp_ratio=4))

    def forward(self, x):
        x = self.conv1(x)
        feat1 = self.mv2[0](x)

        x = self.mv2[1](feat1)
        x = self.mv2[2](x)
        feat2 = self.mv2[3](x)

        x = self.mv2[4](feat2)
        feat3 = self.poolt[0](x)

        x = self.mv2[5](feat3)
        feat4 = self.poolt[1](x)

        x = self.mv2[6](feat4)
        feat5 = self.poolt[2](x)
        return [feat1, feat2, feat3, feat4, feat5]


def encoder_branch_msi():
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 128, 128, 256, 256, 512, 512]
    print('encoder_branch')
    inchannels = 6
    return Encoder((256, 256), dims, channels, inchannels)


def encoder_branch_sar():
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 128, 128, 256, 256, 512, 512]
    print('encoder_branch')
    inchannels = 1
    return Encoder((256, 256), dims, channels, inchannels)


def count_parameters(model, img):
    from thop import clever_format, profile
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_input = torch.randn(1, 6, img.shape[2], img.shape[3]).to(device)
    flops, params = profile(model.to(device), (dummy_input, ), verbose=False)
    # --------------------------------------------------------#
    #   flops * 2是因为profile没有将卷积作为两个operations
    #   有些论文将卷积算乘法、加法两个operations。此时乘2
    #   有些论文只考虑乘法的运算次数，忽略加法。此时不乘2
    #   本代码选择乘2，参考YOLOX。
    # --------------------------------------------------------#
    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))


if __name__ == '__main__':
    img = torch.randn(8, 6, 256, 256)
    vits = encoder_branch_msi()
    out = vits(img)
    vitspara = count_parameters(vits, img)
    # print(vitspara)
    print(out[0].shape)
    print(out[1].shape)
    print(out[2].shape)
    print(out[3].shape)
    print(out[4].shape)
