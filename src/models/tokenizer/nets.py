"""
Credits to https://github.com/CompVis/taming-transformers
"""

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

'''
encoder:
  _target_: models.tokenizer.Encoder
  config:
    _target_: models.tokenizer.EncoderDecoderConfig
    resolution: 64
    in_channels: 3
    z_channels: 512
    ch: 64
    ch_mult: [1, 1, 1, 1, 1]
    num_res_blocks: 2
    attn_resolutions: [8, 16]
    out_ch: 3
    dropout: 0.0
'''
@dataclass
class EncoderDecoderConfig:
    # 输入图像的初始分辨率,（w h）
    resolution: int
    # 输入图像的通道数。RGB 图像，in_channels 通常是 3，而对于灰度图像是 1
    in_channels: int
    # 输出的嵌入特征的通道数。编码器的输出是经过多次下采样后的特征图，这些特征图的通道数被压缩为 z_channels，用于进一步处理或作为解码器的输入
    z_channels: int
    # 基础通道数，用于在特征提取的第一层
    ch: int
    # 通道数的倍增列表，每层的通道数可以不断增加，以捕获更丰富的特征。
    ch_mult: List[int]
    # 每个分辨率下包含的残差块（ResNet Block）数量
    num_res_blocks: int
    # 在哪些分辨率上添加注意力模块
    attn_resolutions: List[int]
    # 输出特征的通道数
    out_ch: int
    # Dropout 概率，用于防止模型过拟合
    dropout: float

    # num_timesteps
    num_timesteps: int = 10


class Encoder(nn.Module):
    def __init__(self, config: EncoderDecoderConfig) -> None:
        super().__init__()
        self.config = config
        # 获取分辨率级别的数量，这决定了编码器的深度。
        self.num_resolutions = len(config.ch_mult)
        # 时间步嵌入通道数（未使用）
        temb_ch = 0  # timestep embedding #channels

        # downsampling
        # 输入卷积层，将输入图像转换为指定数量的通道
        self.conv_in = torch.nn.Conv2d(config.in_channels,
                                       config.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # 当前分辨率（w,h）
        curr_res = config.resolution
        # eg,in_ch_mult = (1, 2, 4, 8)
        in_ch_mult = (1,) + tuple(config.ch_mult)
        # 用于存储下采样层的模块列表
        self.down = nn.ModuleList()
        # 遍历每个分辨率级别
        for i_level in range(self.num_resolutions):
            # 存储残差块和注意力块的模块列表
            block = nn.ModuleList()
            attn = nn.ModuleList()
            # 计算当前块的输入通道数，i_level = 0，block_in = config.ch * 1
            block_in = config.ch * in_ch_mult[i_level]
            # 计算当前块的输出通道数，i_level = 0，block_out = config.ch * 2
            block_out = config.ch * config.ch_mult[i_level]
            # 在每个分辨率级别上添加多个残差块
            # 时间步嵌入的通道数，在某些任务（例如扩散模型）中用于引入时间信息。这里为 0，表示不使用时间嵌入。
            for i_block in range(self.config.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=temb_ch,
                                         dropout=config.dropout))
                block_in = block_out
                # 如果当前分辨率在注意力分辨率列表中，则添加一个注意力模块
                if curr_res in config.attn_resolutions:
                    attn.append(AttnBlock(block_in))
            # 创建一个模块容器来存储每个分辨率级别的块和注意力模块
            down = nn.Module()
            # 将残差块和注意力模块添加到容器中。
            down.block = block
            down.attn = attn
            # 如果不是最后一个分辨率级别，则添加一个下采样层，将分辨率减半
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, with_conv=True)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        # 创建一个 PyTorch 模块 (nn.Module) 作为中间模块的容器，用于存放多个子模块（残差块和注意力块）
        self.mid = nn.Module()
        # out_channels=block_in：输出通道数与输入通道数相同，保持特征维度不变。
        # temb_channels=temb_ch：时间嵌入的通道数，通常用于时间序列任务中；这里为 0，表示不使用时间信息。
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=temb_ch,
                                       dropout=config.dropout)
        # 输入通道数为 block_in，和上一个残差块保持一致
        self.mid.attn_1 = AttnBlock(block_in)
        # 定义了第二个残差块
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=temb_ch,
                                       dropout=config.dropout)
        # 中间模块通常通过残差块 + 注意力块 + 残差块的结构来增强特征的表示能力，同时保留空间和通道上的上下文信息

        # end

        # 对特征进行归一化的层，用于标准化特征，使得数据的均值和方差满足一定的标准。
        self.norm_out = Normalize(block_in)
        # 输出卷积层 (Conv2d)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        config.z_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入是一个四维张量 x（通常是形状为 (batch_size, channels, height, width)(b, 3, 64, 64) 的图像
        # temb = None：时间嵌入，通常在时间相关的任务（例如扩散模型）中使用，这里设置为 None 表示不使用时间嵌入
        temb = None  # timestep embedding

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.config.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        # 获取下采样阶段最后的特征图
        h = hs[-1]
        # 通过第一个残差块，增强局部特征
        h = self.mid.block_1(h, temb)
        # 通过注意力模块，捕获输入数据的全局依赖关系
        h = self.mid.attn_1(h)
        # 通过第二个残次块，结合全局和局部星系，提升特征的表达能力
        h = self.mid.block_2(h, temb)

        # end
        # 对特征进行归一化，以稳定训练过程和提高模型的泛化能力。
        h = self.norm_out(h)
        h = nonlinearity(h)
        # 通过卷积层 conv_out，将特征图的通道数转换为目标通道数，得到最终的编码输出
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(self, config: EncoderDecoderConfig) -> None:
        super().__init__()
        self.config = config
        temb_ch = 0
        # 获取分辨率级别的数量，决定解码器的深度。
        self.num_resolutions = len(config.ch_mult)

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(config.ch_mult)
        # 解码器第一个卷积层的输出通道数。
        block_in = config.ch * config.ch_mult[self.num_resolutions - 1]
        # 计算当前分辨率，从最底层（最小分辨率）开始恢复。
        curr_res = config.resolution // 2 ** (self.num_resolutions - 1)
        # （,,）
        print(f"Tokenizer : shape of latent is {config.z_channels, curr_res, curr_res}.")

        # z to block_in
        self.conv_in = torch.nn.Conv2d(config.z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        # 中间部分由两个 ResNet 块 和一个 注意力块 组成
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=temb_ch,
                                       dropout=config.dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=temb_ch,
                                       dropout=config.dropout)

        # upsampling
        self.up = nn.ModuleList()
        # 从低分辨率向高分辨率逐层上采样，恢复图像的分辨率
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = config.ch * config.ch_mult[i_level]
            # + 1 主要是为了确保在解码器的每个分辨率级别上比编码器多添加一个额外的残差块。这样做的原因是为了增加模型在解码阶段的复杂性，使得特征可以在解码过程中得到更丰富的处理和恢复。
            for i_block in range(config.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=temb_ch,
                                         dropout=config.dropout))
                block_in = block_out
                if curr_res in config.attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, with_conv=True)
                curr_res = curr_res * 2
            # 将上采样模块插入到 self.up 的最前面
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        # 标准化
        self.norm_out = Normalize(block_in)
        # 定义输出卷积层，将特征图从内部通道数 (block_in) 转换到最终的输出通道数 (config.out_ch)
        # (64,64,3)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        config.out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        temb = None  # timestep embedding

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            # 注意多加了一个块
            for i_block in range(self.config.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class DiffusionModelDecoder(nn.Module):
    def __init__(self, config: EncoderDecoderConfig, num_timesteps: int, model_dim: int):
        super(DiffusionModelDecoder, self).__init__()
        # 设置扩散模型的参数
        self.num_timesteps = num_timesteps
        self.model_dim = model_dim

        # 输入应该是(B, 512, 4, 4)
        # 定义UNet结构用于扩散过程中的噪声估计
        self.unet = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 上采样，输出分辨率变为 8x8
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 上采样，输出分辨率变为 16x16
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 上采样，输出分辨率变为 32x32
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 上采样，输出分辨率变为 64x64
            nn.Conv2d(32, 3, 3, padding=1),
        )

        # 定义时间步嵌入层
        self.time_embedding = nn.Embedding(num_timesteps, model_dim)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x 是输入的图像，t 是当前时间步
        # 嵌入时间步并将其与输入数据结合
        t_embedded = self.time_embedding(t).unsqueeze(-1).unsqueeze(-1)
        t_embedded = t_embedded.expand(-1, -1, x.size(-2), x.size(-1))
        x = torch.cat((x, t_embedded), dim=1)
        
        # 经过UNet网络预测噪声
        predicted_noise = self.unet(x)
        return predicted_noise

    def add_noise(self, x: torch.Tensor, t: int) -> torch.Tensor:
        # 添加噪声到输入图像中，模拟扩散过程
        noise = torch.randn_like(x)
        beta = self.get_beta_schedule(t)
        noisy_x = np.sqrt(1 - beta) * x + np.sqrt(beta) * noise
        return noisy_x

    def get_beta_schedule(self, t: int) -> float:
        # 获取时间步 t 对应的噪声强度，beta 通常在扩散过程中是逐渐增大的
        return 0.01 + 0.02 * (t / self.num_timesteps)  # 线性变化，仅作示例

    def denoise(self, noisy_x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 对噪声图像去噪，恢复原始图像
        for timestep in reversed(range(self.num_timesteps)):
            noise_pred = self(noisy_x, t)
            beta = self.get_beta_schedule(timestep)
            noisy_x = (noisy_x - np.sqrt(beta) * noise_pred) / np.sqrt(1 - beta)
        return noisy_x

    def compute_loss(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 计算扩散模型的损失函数
        # Step 1: 向输入图像中添加噪声
        noisy_x = self.add_noise(x, t)
        # Step 2: 预测噪声
        predicted_noise = self(noisy_x, t)
        # Step 3: 计算真实噪声
        real_noise = noisy_x - x
        # Step 4: 使用L2损失计算噪声估计误差
        loss = F.mse_loss(predicted_noise, real_noise)
        return loss



'''
这是 Swish 激活函数的定义，计算过程是将输入张量 x 和它的 sigmoid 激活值逐元素相乘
torch.sigmoid(x):这会将输入值压缩到 (0, 1) 的范围。
x * torch.sigmoid(x)：通过将原始输入和其 sigmoid 激活值相乘，实现了一种“自调节”的非线性激活方式
在特征提取、压缩、生成等需要细致处理输入特征的模型中，Swish 常用于代替 ReLU 以提高模型的表示能力。
'''
def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    # swish
    return x * torch.sigmoid(x)

'''
返回一个 Group Normalization (GroupNorm) 层
参数：
num_groups=32：将输入通道分成 32 组。Group Normalization 的关键是将输入通道划分为若干组，每组内进行均值和方差的归一化处理。32 组通常是一个较为通用的选择，适合大多数特征图。
num_channels=in_channels：指定输入通道的数量，即对多少个通道进行归一化
eps=1e-6：为了防止归一化过程中出现除以零的情况，添加一个非常小的数 (eps) 到方差中，确保计算稳定性
affine=True：如果为 True，则在归一化后会通过可学习的参数进行缩放和偏移。也就是说，归一化的结果将进一步乘以一个缩放系数 (scale) 并加上一个偏移 (shift)，这两个参数是可学习的
'''
def Normalize(in_channels: int) -> nn.Module:
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

# 增大特征图的空间分辨率
class Upsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool) -> None:
        super().__init__()
        #  with_conv：是否在上采样后添加一个卷积操作
        self.with_conv = with_conv
        if self.with_conv:
            # 以调整特征并增强非线性表达能力。
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 对输入特征图 x 进行上采样，将空间分辨率扩大 2 倍。
        # 使用最近邻插值 (nearest) 方法。
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

# 减小特征图的空间分辨率
class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool) -> None:
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.with_conv:
            # 为了使卷积的输出分辨率与预期匹配
            # pad 指定了在不同维度上进行的填充量，顺序为 (左, 右, 上, 下)，所以这里在右边和下边各填充 1 个单位
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            # 使用 3x3 卷积核和步长为 2 对输入特征图进行下采样操作。
            x = self.conv(x)
        else:
            # 这种方式相对简单，只是对局部区域取平均，不能学习到复杂的特征
            # 使用平均池化 (avg_pool2d) 进行下采样，池化窗口大小和步长均为 2，将特征图的分辨率减小一半。
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

'''
conv_shortcut：是否使用卷积进行捷径连接（shortcut
temb_channels：时间嵌入通道数（用于一些任务，例如扩散模型）
'''
class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels: int, out_channels: int = None, conv_shortcut: bool = False,
                 dropout: float, temb_channels: int = 512) -> None:
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        # GroupNorm 归一化
        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        # 如果时间嵌入通道数大于 0，则定义一个线性层 temb_proj，将时间嵌入的特征转换为适合卷积层的维度，用于引入时间信息到特征图中。
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        # 定义第二个归一化层
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        # 定义第二个卷积层，将特征保持在 out_channels 维度上
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        # 定义捷径连接（shortcut）如果输入和输出的通道数不同，需要通过捷径连接来调整维度
        if self.in_channels != self.out_channels:
            # 如果指定使用卷积的捷径连接，定义一个卷积层 (conv_shortcut)
            if self.use_conv_shortcut:
                # 3x3 卷积核，其卷积核的感受野更大，因此可以捕捉更多的空间信息
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            # 否则，定义一个 1x1 的卷积层 (nin_shortcut) 来调整通道数
            else:
                # 1x1 卷积更适合用于快速调整通道数，而不会引入额外的空间特征信息。
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = x
        # 第一归一化、激活，这种结构能够增强模型的表达能力，并使模型更易于训练
        # 与卷积层
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        # 
        if temb is not None:
            # [:, :, None, None]：调整时间步嵌入的形状，使其可以加到四维的特征张量 h 上。这会在特征图的空间维度上进行广播操作。
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        # 第二归一化、激活、丢弃与卷积层
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # 捷径连接处理（调整x的通道数）
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        # 残差连接
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels

        # 归一化层，用于标准化输入特征，通常用来稳定训练，帮助减少不同通道之间的分布差异。
        self.norm = Normalize(in_channels)
        # 通过 1x1 卷积对输入特征图进行变换，以生成 Q、K、V，保留空间维度不变，同时可以在通道方向上进行线性映射
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        # 便于残次链接：通过一个 1x1 卷积层对特征图进行线性映射，保证输出特征的通道数与输入一致。
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 初始化归一化特征图x
        h_ = x
        h_ = self.norm(h_)
        # 生成q,k,v
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        # 获取初始形状参数
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)      # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        # 计算查询和键的相似度
        w_ = torch.bmm(q, k)        # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        # 对注意力权重进行缩放，防止在通道数较大时数值过大，提升训练稳定性。
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        # 通过一个 1x1 卷积层对特征图进行线性映射，保证输出特征的通道数与输入一致。
        h_ = self.proj_out(h_)

        # 残次链接
        return x + h_
