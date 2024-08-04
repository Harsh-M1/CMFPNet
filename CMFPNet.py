import torch
import torch.nn as nn
from LGPB import encoder_branch_msi, encoder_branch_sar
from MAFFAB import MAFFAB


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=2):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
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


class CMFPNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(CMFPNetUp, self).__init__()
        self.conv1 = InvertedResidual(
            in_size, out_size, stride=1, expand_ratio=2)
        self.conv2 = InvertedResidual(
            out_size, out_size, stride=1, expand_ratio=2)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs


class CMFPNet(nn.Module):
    def __init__(self, num_classes=3):
        super(CMFPNet, self).__init__()
        self.mobilepool_msi = encoder_branch_msi()
        self.mobilepool_sar = encoder_branch_sar()
        # in_filters = [224, 448, 896, 1344]
        in_filters = [192, 384, 704, 768]
        self.fusion1 = MAFFAB(32*2, 64)
        self.fusion2 = MAFFAB(64*2, 128)
        self.fusion3 = MAFFAB(128*2, 192)
        self.fusion4 = MAFFAB(256*2, 256)
        self.fusion5 = MAFFAB(512*2, 512)
        out_filters = [64, 128, 256, 512]
        # upsampling
        # 64,64,512
        self.up_concat4 = CMFPNetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = CMFPNetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = CMFPNetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = CMFPNetUp(in_filters[0], out_filters[0])
        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            InvertedResidual(out_filters[0], out_filters[0]),
        )
        self.final = nn.Conv2d(out_filters[0], num_classes, 1)

    def forward(self, inputs):
        msi = inputs[:, :6, :, :]  # 前6个通道为MSI
        sar = inputs[:, 6:7, :, :]  # 最后一个通道为SAR-VV
        [msi1, msi2, msi3, msi4,
         msi5] = self.mobilepool_msi.forward(msi)
        [sar1, sar2, sar3, sar4,
         sar5] = self.mobilepool_sar.forward(sar)
        feat5 = self.fusion5(torch.cat([msi5, sar5], 1))
        feat4 = self.fusion4(torch.cat([msi4, sar4], 1))
        feat3 = self.fusion3(torch.cat([msi3, sar3], 1))
        feat2 = self.fusion2(torch.cat([msi2, sar2], 1))
        feat1 = self.fusion1(torch.cat([msi1, sar1], 1))

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)
        up1 = self.up_conv(up1)

        final = self.final(up1)

        return final


def count_parameters(model, img):
    from thop import clever_format, profile
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_input = torch.randn(
        1, img.shape[1], img.shape[2], img.shape[3]).to(device)
    flops, params = profile(model.to(device), (dummy_input, ), verbose=False)
    flops = flops * 2
    flops, params = clever_format([flops, params], "%.3f")
    print('Total GFLOPS: %s' % (flops))
    print('Total params: %s' % (params))


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = torch.randn(16, 7, 256, 256).to(device)
    model = CMFPNet().to(device)
    para = count_parameters(model, img)
    out = model(img)
    # print(para)
    print(out.shape)
