import torch
import torch.nn as nn
from torch.nn import functional as F
#  Multidimensional Adaptive Frequency Filtering Attention Block


class Conv2dGNGELU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        # 计算填充量以保持空间尺寸（在卷积操作前后尺寸不变，假设步长为1）
        padding = (kernel_size - 1) // 2
        # 调用父类nn.Sequential的初始化函数，并依次添加以下层：
        super().__init__(
            # 2D卷积层，不包含偏置项（bias=False）
            nn.Conv2d(in_channel, out_channel, kernel_size,
                      stride, padding, groups=groups, bias=False),
            # 分组归一化层，这里的group数量设为4（通常是任意选择的，但应与通道数相匹配或为其因子）
            nn.GroupNorm(4, out_channel),
            # GELU激活函数层
            nn.GELU()
        )

    # Conv1dGNGELU类是一个自定义的1D卷积模块，它结合了1D卷积、GroupNorm和GELU激活函数。


class Conv1dGNGELU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        # 计算填充量以保持空间尺寸（在卷积操作前后尺寸不变，假设步长为1）
        padding = (kernel_size - 1) // 2
        # 调用父类nn.Sequential的初始化函数，并依次添加以下层：
        super().__init__(
            # 1D卷积层，不包含偏置项（bias=False）
            nn.Conv1d(in_channel, out_channel, kernel_size,
                      stride, padding, groups=groups, bias=False),
            # 分组归一化层，这里的group数量设为4（通常是任意选择的，但应与通道数相匹配或为其因子）
            nn.GroupNorm(4, out_channel),
            # GELU激活函数层
            nn.GELU()
        )


# InvertedDepthWiseConv2d类是一个自定义的模块，实现了反转深度可分离卷积（Inverted Depthwise Convolution）
class InvertedDepthWiseConv2d(nn.Module):
    # 构造函数
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, expand_ratio=2):
        super().__init__()  # 调用父类nn.Module的初始化函数
        hidden_channel = in_channel * expand_ratio  # 计算扩展后的隐藏通道数
        # 判断是否需要使用快捷连接（shortcut），通常当步长为1且输入输出通道数相同时使用
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []  # 初始化一个空列表，用于存放将要组合成序列的层
        # 添加1x1逐点卷积（pointwise convolution）并跟随GroupNorm和GELU激活函数
        layers.append(Conv2dGNGELU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 添加3x3深度可分离卷积（depthwise convolution），并跟随GroupNorm和GELU激活函数
            Conv2dGNGELU(hidden_channel, hidden_channel,
                         kernel_size=kernel_size, stride=stride, groups=hidden_channel),
            # 添加另一个1x1逐点卷积，但这次不使用激活函数（线性变换）
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            # 添加GroupNorm层，但不添加激活函数，因为可能在后面的操作中与shortcut进行相加
            nn.GroupNorm(4, out_channel),
        ])
        # 将列表中的层组合成一个Sequential序列，这样在前向传播时它们会依次被执行
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class InvertedDepthWiseConv1d(nn.Module):
    # 初始化函数
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, expand_ratio=2):
        super().__init__()  # 调用父类的初始化函数
        hidden_channel = in_channel * expand_ratio  # 计算隐藏层的通道数，它是输入通道数与扩展比例的乘积
        self.use_shortcut = stride == 1 and in_channel == out_channel  # 判断是否使用残差连接（shortcut）

        layers = []  # 创建一个空列表来存储网络层

        # 添加一个1x1的逐点卷积层（不改变空间尺寸，只改变通道数），并使用GELU激活函数
        layers.append(Conv1dGNGELU(in_channel, hidden_channel, kernel_size=1))

        # 扩展layers列表，添加以下层：
        layers.extend([
            # 添加一个3x3的深度可分离卷积层，步长可能改变空间尺寸，并使用GELU激活函数
            Conv1dGNGELU(hidden_channel, hidden_channel,
                         kernel_size=kernel_size, stride=stride, groups=hidden_channel),
            # 添加一个1x1的逐点卷积层（不改变空间尺寸，只改变通道数），不使用激活函数（线性）
            nn.Conv1d(hidden_channel, out_channel, kernel_size=1, bias=False),
            # 添加分组归一化层，这里的group数量设为4（通常是任意选择的，但应与通道数相匹配或为其因子）
            nn.GroupNorm(4, out_channel),
        ])

        # 将所有层串联起来形成一个序列化的模型
        self.conv = nn.Sequential(*layers)

        # 前向传播函数
    def forward(self, x):
        if self.use_shortcut:  # 如果满足使用残差连接的条件
            return x + self.conv(x)  # 则将输入x与卷积后的结果相加（残差连接）
        else:  # 否则
            return self.conv(x)  # 只返回卷积后的结果


#  Multidimensional Adaptive Frequency Filtering Attention Block
class MAFFAB (nn.Module):
    def __init__(self, indim, dim, a=16, b=16, c_h=16, c_w=16):
        super().__init__()
        self.conv1 = InvertedDepthWiseConv2d(indim, dim)
        # 将传入的参数dim、a、b、c_h、c_w注册为不可训练的buffer，并存储为tensor
        self.register_buffer("dim", torch.as_tensor(dim))
        self.register_buffer("a", torch.as_tensor(a))
        self.register_buffer("b", torch.as_tensor(b))
        self.register_buffer("c_h", torch.as_tensor(c_h))
        self.register_buffer("c_w", torch.as_tensor(c_w))

        # 定义可训练的参数a_weight、b_weight、c_weight，并使用ones初始化
        self.a_weight = nn.Parameter(torch.Tensor(2, 1, dim, a))
        nn.init.ones_(self.a_weight)
        self.b_weight = nn.Parameter(torch.Tensor(2, 1, dim, b))
        nn.init.ones_(self.b_weight)
        self.c_weight = nn.Parameter(torch.Tensor(2, dim, c_h, c_w))
        nn.init.ones_(self.c_weight)

        # 定义序列模型wg_a，包含三个Depthwise卷积层，输入和输出通道数均为dim
        self.wg_a = nn.Sequential(
            # 第一层卷积，输入通道数为dim ，输出通道数为2 * dim
            InvertedDepthWiseConv1d(dim, dim),
            # # 第二层卷积，输入和输出通道数均为2 * dim
            # InvertedDepthWiseConv1d(2 * dim, 2 * dim),
            # # 第三层卷积，输入通道数为2 * dim ，输出通道数为dim
            # InvertedDepthWiseConv1d(2 * dim, dim),
        )
        # 定义序列模型wg_b，结构和wg_a相同
        self.wg_b = nn.Sequential(
            # 第一层卷积，输入通道数为dim ，输出通道数为2 * dim
            InvertedDepthWiseConv1d(dim, dim),
            # # 第二层卷积，输入和输出通道数均为2 * dim
            # InvertedDepthWiseConv1d(2 * dim, 2 * dim),
            # # 第三层卷积，输入通道数为2 * dim ，输出通道数为dim
            # InvertedDepthWiseConv1d(2 * dim, dim),
        )

        # 定义序列模型wg_c，这是一个二维卷积模型，结构和wg_a相同
        self.wg_c = nn.Sequential(
            # 第一层卷积，输入和输出通道数均为dim
            InvertedDepthWiseConv2d(dim, dim),
            # # 第二层卷积，输入和输出通道数均为2 * dim
            # InvertedDepthWiseConv2d(2 * dim, 2 * dim),
            # # 第三层卷积，输入通道数为2 * dim ，输出通道数为dim
            # InvertedDepthWiseConv2d(2 * dim, dim),
        )

    def forward(self, input):
        # print("in1",input.shape)
        input = self.conv1(input)
        B, c, a, b = input.size()
        # ----- 对x1进行特殊的卷积操作 ----- #
        # 改变x1的维度顺序以适应后续的傅里叶变换
        # print("in2",input.shape)
        x1 = input.permute(0, 2, 1, 3)
        x1 = torch.fft.rfft2(x1.float(), dim=(2, 3), norm='ortho')
        # 获取a_weight并调整其大小以匹配x1的空间维度
        a_weight = self.a_weight
        a_weight = self.wg_a(F.interpolate(a_weight, size=x1.shape[2:4],
                                           mode='bilinear', align_corners=True
                                           ).squeeze(1)).unsqueeze(1).permute(1, 2, 3, 0)
        a_weight = torch.view_as_complex(a_weight.contiguous())
        x1 = x1 * a_weight
        x1 = torch.fft.irfft2(x1, s=(c, b), dim=(
            2, 3), norm='ortho').permute(0, 2, 1, 3).to(input.dtype)

        # ----- 对x2进行特殊的卷积操作 ----- #
        # 改变x2的维度顺序以适应后续的傅里叶变换
        x2 = input.permute(0, 3, 1, 2)  # B, b, c, a
        # 对x2应用实数傅里叶变换
        x2 = torch.fft.rfft2(x2.float(), dim=(2, 3), norm='ortho')
        # 获取b_weight并调整其大小以匹配x2的空间维度
        b_weight = self.b_weight
        b_weight = self.wg_b(F.interpolate(b_weight, size=x2.shape[2:4],
                                           mode='bilinear', align_corners=True
                                           ).squeeze(1)).unsqueeze(1).permute(1, 2, 3, 0)
        # 将b_weight转换为复数张量
        b_weight = torch.view_as_complex(b_weight.contiguous())
        # 在频域中应用权重
        x2 = x2 * b_weight
        # 应用逆傅里叶变换并将结果转换回原始数据类型
        x2 = torch.fft.irfft2(x2, s=(c, a), dim=(
            2, 3), norm='ortho').permute(0, 2, 3, 1).to(input.dtype)

        # ----- 对x3进行特殊的卷积操作 ----- #
        # 对x3应用实数傅里叶变换
        x3 = torch.fft.rfft2(input.float(), dim=(2, 3), norm='ortho')
        # 获取c_weight并调整其大小以匹配x3的空间维度
        c_weight = self.c_weight
        c_weight = self.wg_c(F.interpolate(c_weight, size=x3.shape[2:4],
                                           mode='bilinear', align_corners=True)).permute(1, 2, 3, 0)
        # 将c_weight转换为复数张量
        c_weight = torch.view_as_complex(c_weight.contiguous())
        # 在频域中应用权重
        x3 = x3 * c_weight
        # 应用逆傅里叶变换并将结果转换回原始数据类型
        x3 = torch.fft.irfft2(x3, s=(a, b), dim=(2, 3),
                              norm='ortho').to(input.dtype)
        # print(x1.shape)
        # print(x2.shape)
        # print(x3.shape)
        x = (x1 + x2 + x3)/3
        return x+input


if __name__ == "__main__":

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 示例数据
    input_tensor = torch.randn(16, 768, 8, 8).to(
        device)  # 生成一个形状为(2, 3, 4, 5)的随机张量
    fftatta = MAFFAB(768, 512, 16, 16, 16, 16).cuda()

    # 前向傅里叶变换和逆变换
    # print(input_tensor)
    x = fftatta(input_tensor)
    # print(fftatta)
    print(x.shape)
