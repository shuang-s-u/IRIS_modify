"""
Credits to https://github.com/karpathy/minGPT
"""

from dataclasses import dataclass
import math
from typing import Optional

from einops import rearrange
import torch
import torch.nn as nn
from torch.nn import functional as F

from .kv_caching import KeysValues, KVCache


@dataclass
class TransformerConfig:
<<<<<<< HEAD
    tokens_per_block: int
    max_blocks: int
    attention: str

    num_layers: int
    num_heads: int
    embed_dim: int

    embed_pdrop: float
    resid_pdrop: float
    attn_pdrop: float

    @property
    def max_tokens(self):
        return self.tokens_per_block * self.max_blocks


=======
    # 每个块可以处理的token数量
    tokens_per_block: int
    # 表示transformer中最大块数量，这决定了transformer的结构中有多少块可以使用
    max_blocks: int
    # 表示自注意力机制的类型
    attention: str

    num_layers: int
    # 注意力头的数量。多头注意力机制可以增强模型的性能
    num_heads: int
    # 嵌入维度，定义了transformer中token的嵌入向量的维度
    embed_dim: int

    # 表示嵌入层的dropout概率
    embed_pdrop: float
    # 表示残次连接的dropout概率
    resid_pdrop: float
    # 表示注意力机制的dropout概率
    attn_pdrop: float

    @property
    # 将 max_tokens 定义为一个只读属性，使得用户可以像访问属性一样调用它，而不需要像方法那样加上括号。
    def max_tokens(self):
        # 表示模型能够处理的最大 token 数。
        return self.tokens_per_block * self.max_blocks

'''
实现了一个典型的 Transformer 模型的结构，包含多个 Block 层、Dropout 和层归一化。
'''
>>>>>>> remotecopy
class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
<<<<<<< HEAD
        self.drop = nn.Dropout(config.embed_pdrop)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)

    def generate_empty_keys_values(self, n: int, max_tokens: int) -> KeysValues:
=======
        # 定义一个 nn.Dropout 层，丢弃率为 config.embed_pdrop，用于防止过拟合。
        # Dropout 是一种正则化技术，通过在训练过程中随机将一部分神经元的输出设置为 0，以减少模型对某些节点的依赖。
        self.drop = nn.Dropout(config.embed_pdrop)
        # 存储多个 Block 实例，数量为 config.num_layers，即 Transformer 的层数。
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        # 定义一个层归一化层 (LayerNorm)，用于对最后的输出进行归一化，维度为 config.embed_dim。
        self.ln_f = nn.LayerNorm(config.embed_dim)

    # 生成空的键和值，用于注意力机制。
    # 参数 n 表示批次大小，max_tokens 表示最大 token 数。
    def generate_empty_keys_values(self, n: int, max_tokens: int) -> KeysValues:
        # 用于获取模型所在的设备（CPU 或 GPU），假设所有子模块都在相同设备上。
>>>>>>> remotecopy
        device = self.ln_f.weight.device  # Assumption that all submodules are on the same device
        return KeysValues(n, self.config.num_heads, max_tokens, self.config.embed_dim, self.config.num_layers, device)

    def forward(self, sequences: torch.Tensor, past_keys_values: Optional[KeysValues] = None) -> torch.Tensor:
<<<<<<< HEAD
        assert past_keys_values is None or len(past_keys_values) == len(self.blocks)
        x = self.drop(sequences)
        for i, block in enumerate(self.blocks):
            x = block(x, None if past_keys_values is None else past_keys_values[i])

=======
        # 确保 past_keys_values 的形状与模型中的层数一致
        assert past_keys_values is None or len(past_keys_values) == len(self.blocks)
        x = self.drop(sequences)
        for i, block in enumerate(self.blocks):
            # x 在每个 block 层中进行处理，输出用于下一层。
            x = block(x, None if past_keys_values is None else past_keys_values[i])

        # 归一化
>>>>>>> remotecopy
        x = self.ln_f(x)
        return x


class Block(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
<<<<<<< HEAD
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.attn = SelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
=======
        # 定义两个嵌入维度为config.embed_dim的归一化层
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        # 定义了一个自注意力层 (SelfAttention)，用于捕捉序列中的全局依赖关系
        self.attn = SelfAttention(config)
        # 定义了一个前馈神经网络 (mlp)，使用 nn.Sequential 将多个层组合在一起
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            # 激活函数，GELU（Gaussian Error Linear Unit），它在深度学习中比 ReLU 更常用，用于增强模型的非线性表示能力。
>>>>>>> remotecopy
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x: torch.Tensor, past_keys_values: Optional[KeysValues] = None) -> torch.Tensor:
<<<<<<< HEAD
        x_attn = self.attn(self.ln1(x), past_keys_values)
=======
        # 进行层归一化（self.ln1(x)），然后将归一化后的结果传入自注意力层 self.attn（B，T，C）
        x_attn = self.attn(self.ln1(x), past_keys_values)
        # 使用残差连接，将自注意力层的输出 (x_attn) 加到原始输入 (x) 上。
        # 残差连接的作用是防止梯度消失，帮助模型训练更深的网络。
>>>>>>> remotecopy
        x = x + x_attn
        x = x + self.mlp(self.ln2(x))
        return x

<<<<<<< HEAD

class SelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        assert config.embed_dim % config.num_heads == 0
        assert config.attention in ('causal', 'block_causal')
        self.num_heads = config.num_heads
        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

        causal_mask = torch.tril(torch.ones(config.max_tokens, config.max_tokens))
        block_causal_mask = torch.max(causal_mask, torch.block_diag(*[torch.ones(config.tokens_per_block, config.tokens_per_block) for _ in range(config.max_blocks)]))
        self.register_buffer('mask', causal_mask if config.attention == 'causal' else block_causal_mask)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        B, T, C = x.size()
        if kv_cache is not None:
            b, nh, L, c = kv_cache.shape
=======
'''
定义了一个名为 SelfAttention 的类，用于实现 Transformer 模型中的自注意力机制。
自注意力是 Transformer 模型的核心部分，用于捕捉输入序列中每个位置之间的依赖关系。
'''
class SelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        # 保证每个头的维度是整数
        assert config.embed_dim % config.num_heads == 0
        # 确保注意力类型是 'causal' 或 'block_causal'，用于设置注意力掩码
        assert config.attention in ('causal', 'block_causal')
        self.num_heads = config.num_heads
        # 定义三个线性层,用于生成注意力机制所需的键、查询和值向量，输入和输出的维度都是 embed_dim
        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)
        # attn_drop 用于对注意力权重进行丢弃，减少过拟合/resid_drop 用于对输出特征进行丢弃
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # 对注意力计算后的输出进行变换
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

        # 创建一个下三角矩阵 causal_mask，用于实现因果（causal）掩码。
        # 这个掩码确保在计算注意力时，每个位置只能看到它之前的位置，以便在生成任务中保持时间顺序。
        causal_mask = torch.tril(torch.ones(config.max_tokens, config.max_tokens))
        # 创建一个块状的因果掩码 block_causal_mask，用于实现分块因果注意力。
        block_causal_mask = torch.max(causal_mask, torch.block_diag(*[torch.ones(config.tokens_per_block, config.tokens_per_block) for _ in range(config.max_blocks)]))
        # 将掩码矩阵（causal_mask 或 block_causal_mask）注册为模型的缓冲区变量，名称为 'mask'
        self.register_buffer('mask', causal_mask if config.attention == 'causal' else block_causal_mask)

    # 主要用于执行自注意力机制的前向传播，结合输入 x 和（可选的）键值缓存 kv_cache 进行计算
    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        # 形状为 (B, T, C)，分别代表批次大小、序列长度和嵌入维度。
        B, T, C = x.size()
        if kv_cache is not None:
            '''
            b 是缓存中的批次大小。
            nh 是注意力头的数量。
            L 是缓存中已处理的序列长度。
            c 是每个头的嵌入维度
            '''
            b, nh, L, c = kv_cache.shape
            # 检查kv_cache的形状是否与输入匹配
>>>>>>> remotecopy
            assert nh == self.num_heads and b == B and c * nh == C
        else:
            L = 0

<<<<<<< HEAD
=======
        # 使用三个线性变换（query、key 和 value）生成查询向量 q、键向量 k 和值向量 v。
        # (B, nh, T, hs) (批次，头大小，序列长度，每个头的嵌入维度)
>>>>>>> remotecopy
        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)   # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)     # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)   # (B, nh, T, hs)

<<<<<<< HEAD
        if kv_cache is not None:
            kv_cache.update(k, v)
            k, v = kv_cache.get()

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[L:L + T, :L + T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = rearrange(y, 'b h t e -> b t (h e)')

=======
        # 更新键和值的缓存
        if kv_cache is not None:
            kv_cache.update(k, v)
            # 获取更新后的键和值。这样在长序列的推理中可以通过缓存来减少重复计算。
            k, v = kv_cache.get()

        # 计算了查询和键之间的相似度，通过点积操作，得到每个查询与所有键的注意力分数
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        '''
        masked_fill(mask, value) 用于将 att 中满足条件的元素替换为 value。
        - self.mask[L:L + T, :L + T] == 0 用于选择掩码矩阵中等于 0 的位置，这些位置需要被掩盖。
        - float('-inf') 是负无穷，用于在 softmax 中使得这些位置的注意力权重为 0。
        '''
        att = att.masked_fill(self.mask[L:L + T, :L + T] == 0, float('-inf'))
        # 对注意力分数进行归一化，使得它们转化为概率分布。dim=-1 表示沿着最后一个维度进行 softmax 操作，这里是沿着键的序列长度方向（即每个查询对应的所有键）。
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        # 注意力权重 att 与值向量 v 进行矩阵乘法，得到上下文表示 y
        # att (B, nh, T, T)  /  v (B, nh, T, hs)输出 y (B, nh, T, hs)
        y = att @ v
        # 使用 einops 的 rearrange 函数对上下文向量 y 进行重排列，目的是将不同注意力头的输出拼接在一起
        # 调整后的形状为 (B, T, nh * hs)，(h e)：将所有注意力头的嵌入维度拼接起来，恢复原始的嵌入维度 C
        y = rearrange(y, 'b h t e -> b t (h e)')
        # 使用一个线性层 proj 对上下文向量 y 进行投影，目的是将拼接后的多头上下文向量重新映射回原始的嵌入维度。
        # 最终输出的 y 形状为 (B, T, C)
>>>>>>> remotecopy
        y = self.resid_drop(self.proj(y))

        return y
