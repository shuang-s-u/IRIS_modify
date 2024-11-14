from typing import Tuple

import numpy as np
import torch


class Cache:
    def __init__(self, num_samples: int, num_heads: int, max_tokens: int, embed_dim: int, device: torch.device) -> None:
        assert embed_dim % num_heads == 0
        self._n, self._cache, self._size = num_samples, None, None
        # 一个匿名函数（lambda function），它接收一个参数 n。n 通常表示批次大小（Batch Size）。
        self._reset = lambda n: torch.empty(n, num_heads, max_tokens, embed_dim // num_heads, device=device)  # (B, nh, T, hs)
        self.reset()

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        n, num_heads, _, head_dim = self._cache.shape
        return n, num_heads, self._size, head_dim

    def reset(self) -> None:
        # 使用 _reset 函数生成新的空缓存张量，并赋值给 _cache
        self._cache = self._reset(self._n)
        # 将 _size 设置为 0，表示缓存还没有存储任何信息
        self._size = 0

    def prune(self, mask: np.ndarray) -> None:
        # 断言 mask 是一个一维数组，并且其长度与缓存的批次大小一致。
        # 这是为了确保掩码可以正确地用于选择批次中的样本。
        assert mask.ndim == 1 and mask.shape[0] == self.shape[0]
        self._cache = self._cache[mask]
        # 裁剪后的缓存中样本的数量
        self._n = self._cache.shape[0]

    def get(self) -> torch.Tensor:
        return self._cache[:, :, :self._size, :]

    def update(self, x: torch.Tensor) -> None:
        # 是确保新输入张量的形状和缓存匹配，除时间步维度
        assert (x.ndim == self._cache.ndim) and all([x.size(i) == self._cache.size(i) for i in (0, 1, 3)])
        # 确保添加新的张量后，缓存的大小不会超出最大限制。
        assert self._size + x.size(2) <= self._cache.shape[2]
        self._cache = AssignWithoutInplaceCheck.apply(self._cache, x, 2, self._size, self._size + x.size(2))
        self._size += x.size(2)


class KVCache:
    def __init__(self, n: int, num_heads: int, max_tokens: int, embed_dim: int, device: torch.device) -> None:
        self._k_cache = Cache(n, num_heads, max_tokens, embed_dim, device)
        self._v_cache = Cache(n, num_heads, max_tokens, embed_dim, device)

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        # 获得缓存的形状
        return self._k_cache.shape

    def reset(self) -> None:
        self._k_cache.reset()
        self._v_cache.reset()

    def prune(self, mask: np.ndarray) -> None:
        self._k_cache.prune(mask)
        self._v_cache.prune(mask)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._k_cache.get(), self._v_cache.get()

    def update(self, k: torch.Tensor, v: torch.Tensor):
        self._k_cache.update(k)
        self._v_cache.update(v)


class KeysValues:
    '''
    n：表示批量大小。
    num_heads：表示 Transformer 中的注意力头的数量。
    max_tokens：最大 token 数量，用于确定每层缓存能够保存的最大 token 数。
    embed_dim：嵌入维度。
    num_layers：Transformer 层数。
    device：指定计算设备（如 CPU 或 GPU）
    '''
    def __init__(self, n: int, num_heads: int, max_tokens: int, embed_dim: int, num_layers: int, device: torch.device) -> None:
        # 这是一个元组，包含多个 KVCache 对象，每个 KVCache 对象用于存储某一层 Transformer 的键和值缓存。
        self._keys_values = tuple([KVCache(n, num_heads, max_tokens, embed_dim, device) for _ in range(num_layers)])
    
    # 使得 KeysValues 对象能够像列表一样被索引
    def __getitem__(self, key: int) -> KVCache:
        return self._keys_values[key]

    # 可以使用 len(kv) 来获取缓存的层数
    def __len__(self):
        return len(self._keys_values)

    @property
    def size(self):
        # 返回键值缓存中每层的时间步数
        return self._keys_values[0].shape[2]

    def reset(self) -> None:
        # 重置每个 KVCache 对象，通常用于清空缓存中的键和值。
        for kv_cache in self._keys_values:
            kv_cache.reset()

    def prune(self, mask: np.ndarray) -> None:
        # 对每个 KVCache 进行剪枝，通常用于选择性地保留某些缓存条目。
        for kv_cache in self._keys_values:
            kv_cache.prune(mask)


class AssignWithoutInplaceCheck(torch.autograd.Function):
    """
    Inspired from : https://discuss.pytorch.org/t/disable-in-place-correctness-version-check-any-other-workaround/90738/4
    Warning : do not use it to overwrite a slice twice.

    AssignWithoutInplaceCheck 继承自 torch.autograd.Function，用于实现自定义的前向和反向传播
    """

    @staticmethod
    def get_slice(dim: int, start: int, stop: int) -> Tuple[slice]:
        # slice(None) 等价于 [:]，表示选择所有元素
        return tuple([slice(None), ] * dim + [slice(start, stop)])

    '''
    input：原始张量，将要被部分更新的数据。
    value：要用来更新 input 特定部分的新值张量。
    dim、start、stop：指定在哪个维度上（dim），从哪里开始（start）到哪里结束（stop）对 input 进行切片，并用 value 进行赋值操作。
    '''
    @staticmethod
    def forward(ctx, input: torch.Tensor, value: torch.Tensor, dim: int, start: int, stop: int) -> torch.Tensor:
        # ctx：上下文对象,用于存储前向传播中保存的信息，帮助反向传播过程。
        ctx.dim = dim
        ctx.start = start
        ctx.stop = stop
        '''
        .data 用于直接访问张量的底层数据，并进行操作。这是为了绕过 PyTorch 内部的 in-place 操作检查，直接修改张量的内容。
        **注意：**这种方式可能存在风险，因为它不会对计算图做版本检查（不推荐直接修改 .data，但在这里用来避免不必要的 in-place 限制
        '''
        # input.data[slice(None), slice(None), slice(4, 7)]
        input.data[AssignWithoutInplaceCheck.get_slice(dim, start, stop)] = value
        return input

    '''
    返回值的结构：返回值是一个元组，包含对前向传播中各输入的梯度。
                由于前向传播中有 5 个输入参数（input, value, dim, start, stop），backward() 需要返回对应的梯度信息，这里我们返回了 5 个值。
    '''
    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor]:
        # 对于 value，前向传播中 value 只被赋值到 input 的某一部分，所以 value 的梯度只涉及到对应部分。使用该切片从 grad_out 中提取对应部分的梯度。
        return grad_out, grad_out[AssignWithoutInplaceCheck.get_slice(ctx.dim, ctx.start, ctx.stop)], None, None, None
'''
在前向传播中，AssignWithoutInplaceCheck 只是在 input 张量的某部分赋值了 value，没有对 input 做任何其他操作（比如函数运算、变换等）。
对于反向传播来说，这意味着损失对 input 的偏导数并不受到赋值操作的影响。实际上，赋值操作只是更改了 input 的某部分的值，其他部分保持不变。
因此，损失函数对 input 的导数直接等于 grad_out，因为 grad_out 本身就是从损失对 input 的梯度，不需要再进行额外的处理。
'''
