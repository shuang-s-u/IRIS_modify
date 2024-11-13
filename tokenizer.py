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
@dataclass
class TokenizerEncoderOutput:
    z: torch.FloatTensor
    z_quantized: torch.FloatTensor
    tokens: torch.LongTensor



class Tokenizer(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, encoder: Encoder, decoder: Decoder, with_lpips: bool = True) -> None:
        super().__init__()
        # vocab_size：词汇表大小，即离散化的编码数量
        self.vocab_size = vocab_size
        self.encoder = encoder
        # 目前config文件中的 z_channels = embed_dim
        self.pre_quant_conv = torch.nn.Conv2d(encoder.config.z_channels, embed_dim, 1)
        # z_channels = embed_dim = vocab_size = 512 
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 对解码器输入进行卷积，使其从嵌入维度恢复到合适的通道数
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, decoder.config.z_channels, 1)
        self.decoder = decoder
        self.embedding.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
        # 感知损失模块，用于计算重构图像与原始图像之间的感知差异。
        self.lpips = LPIPS().eval() if with_lpips else None

        # todo:init dm

    def __repr__(self) -> str:
        return "tokenizer"

    def forward(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> Tuple[torch.Tensor]:
        # outputs，包含原始编码 z，量化后的 z_quantized，以及 tokens
        outputs = self.encode(x, should_preprocess)
        # z_quantized 离散化的表示
        decoder_input = outputs.z + (outputs.z_quantized - outputs.z).detach()
        reconstructions = self.decode(decoder_input, should_postprocess)
        return outputs.z, outputs.z_quantized, reconstructions

    '''
    Tokenizer 类中的 compute_loss() 方法，用于计算模型在一批数据上的损失，包括三个部分：
    承诺损失（commitment loss）、重构损失（reconstruction loss） 和 感知损失（perceptual loss）。这些损失用于指导模型的训练，让模型学习到更有效的表示和更好的重构能力。
    '''
    def compute_loss(self, batch: Batch, **kwargs: Any) -> LossWithIntermediateLosses:
        assert self.lpips is not None

        observations = self.preprocess_input(rearrange(batch['observations'], 'b t c h w -> (b t) c h w'))
        z, z_quantized, reconstructions = self(observations, should_preprocess=False, should_postprocess=False)

        beta = 1.0
        commitment_loss = (z.detach() - z_quantized).pow(2).mean() + beta * (z - z_quantized.detach()).pow(2).mean()

        reconstruction_loss = torch.abs(observations - reconstructions).mean()
        perceptual_loss = torch.mean(self.lpips(observations, reconstructions))

        return LossWithIntermediateLosses(commitment_loss=commitment_loss, reconstruction_loss=reconstruction_loss, perceptual_loss=perceptual_loss)

    def encode(self, x: torch.Tensor, should_preprocess: bool = False) -> TokenizerEncoderOutput:
        if should_preprocess:
            x = self.preprocess_input(x)
        shape = x.shape 
        x = x.view(-1, *shape[-3:])
        z = self.encoder(x)
        

        z = self.pre_quant_conv(z)
        b, e, h, w = z.shape
        #将批次大小、特征图高度、宽度合并成一个维度，每个位置保留 e 个通道（特征）
        z_flattened = rearrange(z, 'b e h w -> (b h w) e')

        # 计算编码特征到所有嵌入向量的距离，欧氏距离
        dist_to_embeddings = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight**2, dim=1) - 2 * torch.matmul(z_flattened, self.embedding.weight.t())

        # 对距离矩阵 dist_to_embeddings 中的每一行找出最小值的索引，以此来确定最接近的嵌入向量的编号
        # (num_positions,)
        tokens = dist_to_embeddings.argmin(dim=-1)
        # self.embedding(tokens) 会从嵌入表 self.embedding 中获取每个索引对应的嵌入向量。(num_positions, e)
        z_q = rearrange(self.embedding(tokens), '(b h w) e -> b e h w', b=b, e=e, h=h, w=w).contiguous()
        # Reshape to original
        z = z.reshape(*shape[:-3], *z.shape[1:])
        z_q = z_q.reshape(*shape[:-3], *z_q.shape[1:]) # (b, e, h, w)(b, 256, 4*4)
        tokens = tokens.reshape(*shape[:-3], -1) # b, k


        return TokenizerEncoderOutput(z, z_q, tokens)

    def decode(self, z_q: torch.Tensor, should_postprocess: bool = False) -> torch.Tensor:
        # z_q：量化后的张量
        shape = z_q.shape  # (..., E, h, w)
        z_q = z_q.view(-1, *shape[-3:])
        # 将量化后的特征 z_q 从嵌入空间转换到适合解码器输入的空间。
        z_q = self.post_quant_conv(z_q)
        rec = self.decoder(z_q)

        #######################################################-----修改的部分------#######################################################
        # todo: 使用LDM增强编码细节

        # LDM 处理潜在表示
        with torch.no_grad():
            # LDM 的采样阶段，使用 DDIM 进行采样
            ddim_steps = 30  # 设置 DDIM 采样步数
            eta = 0.0  # 设置 eta 为 0 表示确定性采样
            model = get_model()
            model.to(rec.device)
            sampler = DDIMSampler(model)
            mask = torch.ones(rec.shape, dtype=torch.float32).to(rec.device)
            # 生成一个与 rec 相同形状的随机张量，值在 [0, 1) 之间
            random_tensor = torch.rand(rec.shape, device=rec.device)

            # 设置一个阈值，比如 0.5，小于 0.5 的部分变为 0，其他部分保持为 1
            threshold = 0.1
            mask[random_tensor < threshold] = 0

            # 设定去掉部分的比例，比如去掉不超过 10% 的 1
            drop_rate = 0.1

            # 在潜在空间进行扩散采样
            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                            conditioning=None,  # 如果是无条件生成，可以为 None
                                            batch_size=rec.size(0),
                                            shape=rec.shape[1:],  # 潜在空间的维度
                                            mask=mask,
                                            x0=rec,
                                            verbose=False,
                                            eta=eta)
            enhanced_z = samples_ddim

        #将解码后的特征张量 rec 重新调整为与原始输入相兼容的形状
        enhanced_z = enhanced_z.reshape(*shape[:-3], *enhanced_z.shape[1:])


        # rec = rec.reshape(*shape[:-3], *rec.shape[1:]) 

        if should_postprocess:
            # 将张量的值范围从 [-1, 1] 转换为 [0, 1]，适合最终可视化或评估
            # rec = self.postprocess_output(rec)
            rec = self.postprocess_output(enhanced_z)

#######################################################-----end-----#######################################################

        return rec

    @torch.no_grad()
    # 用于推理阶段
    def encode_decode(self, x: torch.Tensor, should_preprocess: bool = False, should_postprocess: bool = False) -> torch.Tensor:
        # import pdb; pdb.set_trace()
        z_q = self.encode(x, should_preprocess).z_quantized
        return self.decode(z_q, should_postprocess)

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """x is supposed to be channels first and in [0, 1]"""
        return x.mul(2).sub(1)

    def postprocess_output(self, y: torch.Tensor) -> torch.Tensor:
        """y is supposed to be channels first and in [-1, 1]"""
        return y.add(1).div(2)
