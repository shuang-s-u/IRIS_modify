import math
from typing import List

import torch
import torch.nn as nn

<<<<<<< HEAD

class Slicer(nn.Module):
    def __init__(self, max_blocks: int, block_mask: torch.Tensor) -> None:
        super().__init__()
        self.block_size = block_mask.size(0)
        self.num_kept_tokens = block_mask.sum().long().item()
        kept_indices = torch.where(block_mask)[0].repeat(max_blocks)
        offsets = torch.arange(max_blocks).repeat_interleave(self.num_kept_tokens)
        self.register_buffer('indices', kept_indices + block_mask.size(0) * offsets)

    def compute_slice(self, num_steps: int, prev_steps: int = 0) -> torch.Tensor:
        total_steps = num_steps + prev_steps
        num_blocks = math.ceil(total_steps / self.block_size)
        indices = self.indices[:num_blocks * self.num_kept_tokens]
        return indices[torch.logical_and(prev_steps <= indices, indices < total_steps)] - prev_steps

    def forward(self, *args, **kwargs):
=======
'''
主要功能：Slicer 类的作用是根据给定的 block_mask 计算出模型前向传播时需要处理的 token 索引。
它根据最大块数（max_blocks）和掩码（block_mask）来动态调整输入 token 的索引位置，使得模型可以有选择性地处理某些 token，而不必处理全部 token。

'''
class Slicer(nn.Module):
    def __init__(self, max_blocks: int, block_mask: torch.Tensor) -> None:
        super().__init__()
        # 获取张量block_mask第一个维度的大小
        self.block_size = block_mask.size(0)
        # 计算 block_mask 中值为 True 的元素数量（即需要保留的 token 数量）
        self.num_kept_tokens = block_mask.sum().long().item()
        # 如果 block_mask 是 torch.tensor([True, False, True, False, True])，那么 torch.where(block_mask) 会返回 (tensor([0, 2, 4]),)。[0] 的作用是从这个元组中提取第一个张量，即需要保留的元素的索引。上述例子中，结果会是 tensor([0, 2, 4])。
        # 将这些索引重复 max_blocks 次，表示这些需要保留的 token 在每个块中都相同。
        kept_indices = torch.where(block_mask)[0].repeat(max_blocks)

        # 为每个块创建一个偏移量，用来标记每个块中 token 的相对位置。
        # 偏移量结合 kept_indices，确保模型在处理多个块时，能够准确地识别出每个 token 的全局位置
        offsets = torch.arange(max_blocks).repeat_interleave(self.num_kept_tokens)
        # 注册不参与梯度更新的tensor，将这些tensor保存为模型的一部分，但不会被反向传播。保留 token 的全局索引。
        self.register_buffer('indices', kept_indices + block_mask.size(0) * offsets)

    '''
    主要功能：返回在当前时间步数范围内，需要处理的保留 token 的相对索引。
    '''
    def compute_slice(self, num_steps: int, prev_steps: int = 0) -> torch.Tensor:
        # 计算总的步数，以便于跟踪当前正在处理的整体进度
        total_steps = num_steps + prev_steps
        # self.block_size 是初始化时根据 block_mask.size(0) 得到的一个块的大小
        # 向上取整，确定需要处理的块的数量。
        num_blocks = math.ceil(total_steps / self.block_size)
        # 提取出所有在当前块数范围内需要处理的 token 的索引。
        indices = self.indices[:num_blocks * self.num_kept_tokens]
        # 返回相对于当前步骤的位置
        return indices[torch.logical_and(prev_steps <= indices, indices < total_steps)] - prev_steps

    def forward(self, *args, **kwargs):
        # 这表明 Slicer 类不是直接用于前向传播的模块，它的主要功能是作为辅助工具，计算保留 token 的索引
>>>>>>> remotecopy
        raise NotImplementedError


class Head(Slicer):
<<<<<<< HEAD
    def __init__(self, max_blocks: int, block_mask: torch.Tensor, head_module: nn.Module) -> None:
        super().__init__(max_blocks, block_mask)
=======
    # head_module：这是一个 nn.Module，表示要应用到输入张量的神经网络模块，用于对提取的部分进行进一步处理。
    def __init__(self, max_blocks: int, block_mask: torch.Tensor, head_module: nn.Module) -> None:
        super().__init__(max_blocks, block_mask)
        # 检查 head_module 是否为 nn.Module 的实例，以确保传入的是有效的 PyTorch 模块。
>>>>>>> remotecopy
        assert isinstance(head_module, nn.Module)
        self.head_module = head_module

    def forward(self, x: torch.Tensor, num_steps: int, prev_steps: int) -> torch.Tensor:
<<<<<<< HEAD
=======
        # 提取这些索引对应的张量部分。
>>>>>>> remotecopy
        x_sliced = x[:, self.compute_slice(num_steps, prev_steps)]  # x is (B, T, E)
        return self.head_module(x_sliced)


<<<<<<< HEAD
class Embedder(nn.Module):
    def __init__(self, max_blocks: int, block_masks: List[torch.Tensor], embedding_tables: List[nn.Embedding]) -> None:
        super().__init__()
        assert len(block_masks) == len(embedding_tables)
        assert (sum(block_masks) == 1).all()  # block mask are a partition of a block
        self.embedding_dim = embedding_tables[0].embedding_dim
        assert all([e.embedding_dim == self.embedding_dim for e in embedding_tables])
        self.embedding_tables = embedding_tables
        self.slicers = [Slicer(max_blocks, block_mask) for block_mask in block_masks]

    def forward(self, tokens: torch.Tensor, num_steps: int, prev_steps: int) -> torch.Tensor:
        assert tokens.ndim == 2  # x is (B, T)
        output = torch.zeros(*tokens.size(), self.embedding_dim, device=tokens.device)
        for slicer, emb in zip(self.slicers, self.embedding_tables):
            s = slicer.compute_slice(num_steps, prev_steps)
            output[:, s] = emb(tokens[:, s])
=======
'''
主要功能：
这段代码定义了一个 Embedder 类，该类的目的是根据不同的 block_mask 将输入的 token 进行划分，
并将这些 token 通过相应的嵌入表（embedding table）转换成嵌入表示。

max_blocks：表示划分的块数，用于创建 Slicer。
block_masks：一个包含 torch.Tensor 的列表，每个 block_mask 表示一个块（block）的模式，用于区分输入中的不同部分（如动作和观察值）。
embedding_tables：一个包含多个 nn.Embedding 的列表，用于不同块的 token 嵌入。
'''
class Embedder(nn.Module):
    def __init__(self, max_blocks: int, block_masks: List[torch.Tensor], embedding_tables: List[nn.Embedding]) -> None:
        super().__init__()
        # 检查，确保每个block_mask都有对应的嵌入网络
        assert len(block_masks) == len(embedding_tables)
        # 确保 block_masks 中的所有掩码都是分区（partition），即它们对每个 token 的划分是相互独立且完整的
        assert (sum(block_masks) == 1).all()  # block mask are a partition of a block
        # 将 embedding_tables 中第一个嵌入表的嵌入维度赋值给 self.embedding_dim，从而为整个 Embedder 对象设置统一的嵌入维度。
        self.embedding_dim = embedding_tables[0].embedding_dim

        # 检查所有的嵌入表的维度是否一致。
        assert all([e.embedding_dim == self.embedding_dim for e in embedding_tables])
        self.embedding_tables = embedding_tables
        # 为每个block_mask都生成一个slicer
        self.slicers = [Slicer(max_blocks, block_mask) for block_mask in block_masks]

    def forward(self, tokens: torch.Tensor, num_steps: int, prev_steps: int) -> torch.Tensor:
        # 检查输入 tokens 的维度必须是 2，（批次大小，序列长度/时间步数）
        assert tokens.ndim == 2  # x is (B, T)
        # 创建一个空的张量 output，用于存储嵌入后的结果。output 的大小是 (B, T, self.embedding_dim)
        output = torch.zeros(*tokens.size(), self.embedding_dim, device=tokens.device)
        # 通过逐步填充 output，Embedder 可以确保所有 token 的嵌入表示被正确计算和保存。
        for slicer, emb in zip(self.slicers, self.embedding_tables):
            s = slicer.compute_slice(num_steps, prev_steps)
            # 选择出特定位置的 token，然后通过嵌入表 emb 进行嵌入。
            # emb(tokens[:, s]) 计算出这些 token 的嵌入表示，大小为 (B, len(s), 128)，即批次中每个样本的 len(s) 个 token 被转换为嵌入表示。
            output[:, s] = emb(tokens[:, s])
            '''
            通过这种方式，output 最终包含了整个序列的嵌入表示，但每个位置的嵌入可能来自不同的嵌入表，具体取决于它们属于哪个块（由 block_masks 决定）。
            这种灵活的处理方式在处理复杂的输入序列（例如包含不同类型的观察值和动作）时非常有用。
            '''
>>>>>>> remotecopy
        return output
