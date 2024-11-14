"""
Credits to https://github.com/CompVis/taming-transformers
"""

from dataclasses import dataclass
from typing import Any, Tuple

from einops import rearrange
import torch
import torch.nn as nn

from dataset import Batch
from .lpips import LPIPS
from .nets import Encoder, Decoder
<<<<<<< HEAD
from utils import LossWithIntermediateLosses


=======
# 
from .latent_diffusion.ldm_iris import get_model
from .latent_diffusion.ldm.models.diffusion.ddim import DDIMSampler
# 
from utils import LossWithIntermediateLosses

'''
TokenizerEncoderOutput 是一个数据类，用于存储 Tokenizer 的编码输出，包括：
z：编码器的原始输出。
z_quantized：量化后的表示（用于重构）。
tokens：每个位置对应的量化编码（token）
'''
>>>>>>> remotecopy
@dataclass
class TokenizerEncoderOutput:
    z: torch.FloatTensor
    z_quantized: torch.FloatTensor
    tokens: torch.LongTensor


<<<<<<< HEAD
class Tokenizer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, encoder: Encoder, decoder: Decoder, with_lpips: bool = True) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = encoder
        self.pre_quant_conv = torch.nn.Conv2d(encoder.config.z_channels, embed_dim, 1)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, decoder.config.z_channels, 1)
        self.decoder = decoder
        self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
        self.lpips = LPIPS().eval() if with_lpips else None

=======

class Tokenizer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, encoder: Encoder, decoder: Decoder, with_lpips: bool = True) -> None:
        super().__init__()
        # vocab_size：词汇表大小，即离散化的编码数量
        self.vocab_size = vocab_size
        self.encoder = encoder
        # 对编码器输出进行卷积，使其匹配嵌入的维度。
        # 1：1*1卷积核
        # 目前config文件中的 z_channels = embed_dim
        self.pre_quant_conv = torch.nn.Conv2d(encoder.config.z_channels, embed_dim, 1)
        # 离散化编码的嵌入表,将离散的 token（整数索引）映射到一个连续的向量空间。
        # z_channels = embed_dim = vocab_size = 512 embedding矩阵是512*512
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 对解码器输入进行卷积，使其从嵌入维度恢复到合适的通道数
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, decoder.config.z_channels, 1)
        self.decoder = decoder
        # self.embedding.weight.data 提供了对权重的直接访问，用于在网络开始训练之前对权重进行初始化。
        # 使用 -1.0 / vocab_size 和 1.0 / vocab_size 作为范围，可以确保初始化值与词汇量的大小相关。
        self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
        # 感知损失模块，用于计算重构图像与原始图像之间的感知差异。
        self.lpips = LPIPS().eval() if with_lpips else None

        # todo:init dm

>>>>>>> remotecopy
    def __repr__(self) -> str:
        return "tokenizer"

    def forward(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> Tuple[torch.Tensor]:
<<<<<<< HEAD
        outputs = self.encode(x, should_preprocess)
=======
        # outputs，包含原始编码 z，量化后的 z_quantized，以及 tokens
        outputs = self.encode(x, should_preprocess)
        # z_quantized 是一种离散化的表示，是从固定的嵌入字典中选取的向量
        # 梯度计算被截断（即 detach()），这意味着在反向传播中，模型只会学习如何让 z 更接近 z_quantized，但不会直接影响 z_quantized。
        # outputs.z_quantized - outputs.z: 计算了量化误差
        # detach() 会生成一个新张量，这个张量与原始张量共享数据，但在反向传播时不参与梯度计算。这意味着对于这部分误差，不会有反向传播对量化后的特征产生梯度影响。
        # 通过构造解码器的输入，使得解码器既能接收量化后的特征，又能通过梯度传播学习如何使编码特征更接近量化特征。
>>>>>>> remotecopy
        decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()
        reconstructions = self.decode(decoder_input, should_postprocess)
        return outputs.z, outputs.z_quantized, reconstructions

<<<<<<< HEAD
    def compute_loss(self, batch: Batch, **kwargs: Any) -> LossWithIntermediateLosses:
        assert self.lpips is not None
        observations = self.preprocess_input(rearrange(batch['observations'], 'b t c h w -> (b t) c h w'))
=======
    '''
    Tokenizer 类中的 compute_loss() 方法，用于计算模型在一批数据上的损失，包括三个部分：
    承诺损失（commitment loss）、重构损失（reconstruction loss） 和 感知损失（perceptual loss）。这些损失用于指导模型的训练，让模型学习到更有效的表示和更好的重构能力。
    '''
    def compute_loss(self, batch: Batch, **kwargs: Any) -> LossWithIntermediateLosses:
        assert self.lpips is not None
        # batch['observations'] 是批次数据中的观测信息
        # 使用 rearrange 重新排列维度，形状从 (b, t, c, h, w) 变为 (b*t, c, h, w)，将时间维度 t 和批次大小 b 合并，使其更方便后续的操作
        observations = self.preprocess_input(rearrange(batch['observations'], 'b t c h w -> (b t) c h w'))
        # 使用 forward() 方法进行前向传播
>>>>>>> remotecopy
        z, z_quantized, reconstructions = self(observations, should_preprocess=False, should_postprocess=False)

        # Codebook loss. Notes:
        # - beta position is different from taming and identical to original VQVAE paper
        # - VQVAE uses 0.25 by default
<<<<<<< HEAD
        beta = 1.0
        commitment_loss = (z.detach() - z_quantized).pow(2).mean() + beta * (z - z_quantized.detach()).pow(2).mean()

        reconstruction_loss = torch.abs(observations - reconstructions).mean()
=======
        # loss1 - 承诺损失的作用是约束编码器输出和离散表示的距离，从而鼓励编码器生成适合量化的表示。
        beta = 1.0
        # 这种双向的约束机制确保了编码器的输出和量化向量（codebook）之间的距离尽量小，从而提高了量化过程的有效性
        commitment_loss = (z.detach() - z_quantized).pow(2).mean() + beta * (z - z_quantized.detach()).pow(2).mean()

        # loss2 - 重构损失用于衡量重构图像和输入图像之间的差异
        # 使用的是 L1 范数（即绝对差异的均值）
        reconstruction_loss = torch.abs(observations - reconstructions).mean()

        # loss3 - 感知损失衡量重构和输入之间的视觉相似性
>>>>>>> remotecopy
        perceptual_loss = torch.mean(self.lpips(observations, reconstructions))

        return LossWithIntermediateLosses(commitment_loss=commitment_loss, reconstruction_loss=reconstruction_loss, perceptual_loss=perceptual_loss)

    def encode(self, x: torch.Tensor, should_preprocess: bool = False) -> TokenizerEncoderOutput:
        if should_preprocess:
            x = self.preprocess_input(x)
<<<<<<< HEAD
        shape = x.shape  # (..., C, H, W)
        x = x.view(-1, *shape[-3:])
        z = self.encoder(x)
        z = self.pre_quant_conv(z)
        b, e, h, w = z.shape
        z_flattened = rearrange(z, 'b e h w -> (b h w) e')
        dist_to_embeddings = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z_flattened, self.embedding.weight.t())

        tokens = dist_to_embeddings.argmin(dim=-1)
        z_q = rearrange(self.embedding(tokens), '(b h w) e -> b e h w', b=b, e=e, h=h, w=w).contiguous()

        # Reshape to original
        z = z.reshape(*shape[:-3], *z.shape[1:])
        z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:])
        tokens = tokens.reshape(*shape[:-3], -1)
=======
        shape = x.shape  # (..., C, H, W) (batch_size, channels, height, width)
        # shape[-3:] 取 shape 的最后三个维度，即 (channels, height, width)
        # -1：告诉 view() 函数，让其自动推导出合适的大小，使总元素个数保持不变。
        # *shape[-3:]解包，最后三个维度保持不变
        x = x.view(-1, *shape[-3:])

        # LDM 处理潜在表示
        with torch.no_grad():
            # LDM 的采样阶段，使用 DDIM 进行采样
            ddim_steps = 30  # 设置 DDIM 采样步数
            eta = 0.0  # 设置 eta 为 0 表示确定性采样
            model = get_model()
            model.to(x.device)
            sampler = DDIMSampler(model)
            mask = torch.ones(x.shape, dtype=torch.float32).to(x.device)
            # 生成一个与 x 相同形状的随机张量，值在 [0, 1) 之间
            random_tensor = torch.rand(x.shape, device=x.device)

            # 设置一个阈值，比如 0.5，小于 0.5 的部分变为 0，其他部分保持为 1
            threshold = 0.1
            mask[random_tensor < threshold] = 0

            # 设定去掉部分的比例，比如去掉不超过 10% 的 1
            drop_rate = 0.1

            # 在潜在空间进行扩散采样
            print(f"rendering {x.size(0)}  in {ddim_steps} steps.")
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                            conditioning=None,  # 如果是无条件生成，可以为 None
                                            batch_size=x.size(0),
                                            shape=x.shape[1:],  # 潜在空间的维度
                                            quantize_x0=True,
                                            mask=mask,
                                            x0=x,
                                            verbose=False,
                                            eta=eta)
            enhanced_x = samples_ddim

        #将解码后的特征张量 x 重新调整为与原始输入相兼容的形状
        enhanced_x = enhanced_x.reshape(*shape[:-3], *enhanced_x.shape[1:])

        # print("Original shape:",shape)
        # print("New shape:",x.shape)

        # print('x', x)
        # print('x', enhanced_x)
        z = self.encoder(enhanced_x)
        # z = self.encoder(x)
        



        #使用卷积层将编码后的特征映射到合适的嵌入维度，通常是为了匹配后续的嵌入操作
        z = self.pre_quant_conv(z)
        # 记录编码输出的形状
        b, e, h, w = z.shape
        #将批次大小、特征图高度、宽度合并成一个维度，每个位置保留 e 个通道（特征）
        z_flattened = rearrange(z, 'b e h w -> (b h w) e')

        # enhanced_z = self.pre_quant_conv(enhanced_z)
        # b, e, h, w = enhanced_z.shape
        # z_flattened = rearrange(enhanced_z, 'b e h w -> (b h w) e')


        # 计算编码特征到所有嵌入向量的距离，欧氏距离
        # dim=1：指定在第 1 个维度上（即特征维度）进行求和操作
        # t()转置操作。
        # z_flattened，(num_positions, e)  /  self.embedding.weight， (vocab_size, e)  / 输出dist_to_embeddings 的形状是 (num_positions, vocab_size)，每个元素表示一个编码特征与某个嵌入向量的距离平方。
        dist_to_embeddings = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z_flattened, self.embedding.weight.t())

        # 对距离矩阵 dist_to_embeddings 中的每一行找出最小值的索引，以此来确定最接近的嵌入向量的编号
        # (num_positions,)
        tokens = dist_to_embeddings.argmin(dim=-1)
        # self.embedding(tokens) 会从嵌入表 self.embedding 中获取每个索引对应的嵌入向量。(num_positions, e)
        # 重排后的形状是 (b, e, h, w)
        # contiguous() 在内存中重新排列张量，使其成为一个连续的张量
        z_q = rearrange(self.embedding(tokens), '(b h w) e -> b e h w', b=b, e=e, h=h, w=w).contiguous()
        # Reshape to original，
        # *shape[:-3]，掉最后三个维度，保留其他所有维度。并解包
        # z.shape[1:] 表示张量 z 除了第一个维度之外的其他所有维度。
        z = z.reshape(*shape[:-3], *z.shape[1:])
        z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:]) # (b, e, h, w)(b, 256, 4*4)
        tokens = tokens.reshape(*shape[:-3], -1) # b, k
        print('IRIS--------------------------------------------------------------------------------------------tokenizer')
        print('z.shape:', z.shape)
        print('z_q.shape', z_q.shape)
        print('tokens', tokens.shape)
>>>>>>> remotecopy

        return TokenizerEncoderOutput(z, z_q, tokens)

    def decode(self, z_q: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
<<<<<<< HEAD
        shape = z_q.shape  # (..., E, h, w)
        z_q = z_q.view(-1, *shape[-3:])
        z_q = self.post_quant_conv(z_q)
        rec = self.decoder(z_q)
        rec = rec.reshape(*shape[:-3], *rec.shape[1:])
        if should_postprocess:
            rec = self.postprocess_output(rec)
        return rec

    @torch.no_grad()
    def encode_decode(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> torch.Tensor:
=======
        # z_q：量化后的张量，通常是编码器输出的量化特征，形状为 (..., E, h, w)
        shape = z_q.shape  # (..., E, h, w)
        # 将所有的批次或时间维度展平，使得后续的卷积操作可以统一处理所有的特征图。
        z_q = z_q.view(-1, *shape[-3:])
        # 将量化后的特征 z_q 从嵌入空间转换到适合解码器输入的空间。
        z_q = self.post_quant_conv(z_q)
        # z_q = self.ldm.decode(z_q)
        rec = self.decoder(z_q) 
        # print(rec) 
        
        # todo: 使用LDM增强编码细节

        # # LDM 处理潜在表示
        # with torch.no_grad():
        #     # LDM 的采样阶段，使用 DDIM 进行采样
        #     ddim_steps = 30  # 设置 DDIM 采样步数
        #     eta = 0.0  # 设置 eta 为 0 表示确定性采样
        #     model = get_model()
        #     model.to(rec.device)
        #     sampler = DDIMSampler(model)
        #     mask = torch.ones(rec.shape, dtype=torch.float32).to(rec.device)
        #     # 生成一个与 rec 相同形状的随机张量，值在 [0, 1) 之间
        #     random_tensor = torch.rand(rec.shape, device=rec.device)

        #     # 设置一个阈值，比如 0.5，小于 0.5 的部分变为 0，其他部分保持为 1
        #     threshold = 0.1
        #     mask[random_tensor < threshold] = 0

        #     # 设定去掉部分的比例，比如去掉不超过 10% 的 1
        #     drop_rate = 0.1

        #     # 在潜在空间进行扩散采样
        #     print(f"rendering {rec.size(0)}  in {ddim_steps} steps.")
        #     samples_ddim, _ = sampler.sample(S=ddim_steps,
        #                                     conditioning=None,  # 如果是无条件生成，可以为 None
        #                                     batch_size=rec.size(0),
        #                                     shape=rec.shape[1:],  # 潜在空间的维度
        #                                     mask=mask,
        #                                     x0=rec,
        #                                     verbose=False,
        #                                     eta=eta)
        #     enhanced_z = samples_ddim

        # #将解码后的特征张量 rec 重新调整为与原始输入相兼容的形状
        # enhanced_z = enhanced_z.reshape(*shape[:-3], *enhanced_z.shape[1:])
        # enhanced_z = torch.clamp((enhanced_z + 1.0)/2.0, min=0.0, max=1.0)
        # print('enhanced_z.shape:', enhanced_z.shape)
        # print(enhanced_z)

        rec = rec.reshape(*shape[:-3], *rec.shape[1:])


        if should_postprocess:
            # 将张量的值范围从 [-1, 1] 转换为 [0, 1]，适合最终可视化或评估
            rec = self.postprocess_output(rec)
            # rec = self.postprocess_output(enhanced_z)

        # difussion model enhance
        # todo: handle input/output shape
        # enhanced_rec = self.dm(rec)

        return rec

    @torch.no_grad()
    # 用于推理阶段，例如当你想要使用编码-解码过程重建图像，而不需要进行反向传播更新模型参数时。
    def encode_decode(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> torch.Tensor:
        # import pdb; pdb.set_trace()
>>>>>>> remotecopy
        z_q = self.encode(x, should_preprocess).z_quantized
        return self.decode(z_q, should_postprocess)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """x is supposed to be channels first and in [0, 1]"""
<<<<<<< HEAD
=======
        # 图像数据：形状为 (batch_size, channels, height, width)
        # 图像数据通常表示为像素值，每个像素的范围可能是 [0, 255]，表示为 8 位整数。如果将其归一化为 [0, 1]，则每个像素的值将是浮点数，范围为 0 到 1。
        # 将输入张量的值从 [0, 1] 变为 [0, 2]，再减去1，使范围变为 [-1, 1]
>>>>>>> remotecopy
        return x.mul(2).sub(1)

    def postprocess_output(self, y: torch.Tensor) -> torch.Tensor:
        """y is supposed to be channels first and in [-1, 1]"""
<<<<<<< HEAD
=======
        # y，表示模型生成的输出图像，形状也是 (batch_size, channels, height, width)
        # 将每个值加上 1，使范围从 [-1, 1] 变为 [0, 2]。将每个值除以 2，使范围变为 [0, 1]。
>>>>>>> remotecopy
        return y.add(1).div(2)
