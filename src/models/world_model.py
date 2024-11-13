from dataclasses import dataclass
from typing import Any, Optional, Tuple

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Batch
from .kv_caching import KeysValues
from .slicer import Embedder, Head
from .tokenizer import Tokenizer
from .transformer import Transformer, TransformerConfig
from utils import init_weights, LossWithIntermediateLosses

'''
@dataclass：这是 Python 的装饰器，用于简化类的定义，特别是那些主要用于存储数据的类。
使用 @dataclass 可以自动生成初始化方法（__init__）、表示方法（__repr__）、比较方法等。
'''
@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    logits_rewards: torch.FloatTensor
    logits_ends: torch.FloatTensor


class WorldModel(nn.Module):    
    # 该类的初始化方法接收观察值的词汇大小 (obs_vocab_size)、
    # 动作的词汇大小 (act_vocab_size)
    # 以及一个 TransformerConfig 类型的配置对象。TransformerConfig 定义了模型的各种配置参数，比如嵌入维度、最大token数量、块的数量等
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, config: TransformerConfig) -> None:
        super().__init__()
        self.obs_vocab_size, self.act_vocab_size = obs_vocab_size, act_vocab_size
        self.config = config
        # 实例化一个transformer对象
        self.transformer = Transformer(config)

        # 定义token模式
        all_but_last_obs_tokens_pattern = torch.ones(config.tokens_per_block)
        # 将倒数第二个观察的token设置为0，其余都为1
        all_but_last_obs_tokens_pattern[-2] = 0
        act_tokens_pattern = torch.zeros(self.config.tokens_per_block)
        # 将倒数第一个token设置为1，其余都为0
        act_tokens_pattern[-1] = 1
        obs_tokens_pattern = 1 - act_tokens_pattern

        ''''
        定义了一个 位置嵌入层，用于为每个序列中的 token 位置生成一个嵌入向量。
        这个向量表示了序列中每个位置的信息，以便在 Transformer 模型中使用时，模型可以感知到输入的时序信息。
        '''
        # 意味着该嵌入层能够为最多 max_tokens 个不同的位置生成对应的嵌入表示
        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)

        self.embedder = Embedder(
            max_blocks=config.max_blocks,
            block_masks=[act_tokens_pattern, obs_tokens_pattern],
            embedding_tables=nn.ModuleList([nn.Embedding(act_vocab_size, config.embed_dim), nn.Embedding(obs_vocab_size, config.embed_dim)])
        )

        self.head_observations = Head(
            max_blocks=config.max_blocks,
            block_mask=all_but_last_obs_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, obs_vocab_size)
            )
        )

        self.head_rewards = Head(
            max_blocks=config.max_blocks,
            block_mask=act_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, 3)
            )
        )

        self.head_ends = Head(
            max_blocks=config.max_blocks,
            block_mask=act_tokens_pattern,
            head_module=nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, 2)
            )
        )

        # 初始化模型中的所有层的权重
        self.apply(init_weights)

    # 打印模型时，会显示出这个字符串
    def __repr__(self) -> str:
        return "world_model"

    def forward(self, tokens: torch.LongTensor, past_keys_values: Optional[KeysValues] = None) -> WorldModelOutput:

        num_steps = tokens.size(1)  # (B, T)
        assert num_steps <= self.config.max_tokens
        prev_steps = 0 if past_keys_values is None else past_keys_values.size

        # 会将 (num_steps, embedding_dim) 自动扩展为形状 (batch_size, num_steps, embedding_dim)，使得加法可以正常进行。
        sequences = self.embedder(tokens, num_steps, prev_steps) + self.pos_emb(prev_steps + torch.arange(num_steps, device=tokens.device))

        # 将嵌入后的序列传入 Transformer 模块进行处理，以学习复杂的时间依赖关系，生成新的表示 x。
        # past_keys_values 可以帮助 Transformer 记住之前的状态，用于高效推理。
        x = self.transformer(sequences, past_keys_values)

        # 计算各个输出头的结果
        logits_observations = self.head_observations(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_rewards = self.head_rewards(x, num_steps=num_steps, prev_steps=prev_steps)
        logits_ends = self.head_ends(x, num_steps=num_steps, prev_steps=prev_steps)

        # 返回最终的输出
        return WorldModelOutput(x, logits_observations, logits_rewards, logits_ends)

    '''
    接收 batch（批量数据）、
    tokenizer（用于编码数据的分词器）
    和其他参数（用 **kwargs 表示任意数量的关键字参数
    返回类型为 LossWithIntermediateLosses，表示返回一个包含多个损失值的对象
    '''
    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, **kwargs: Any) -> LossWithIntermediateLosses:

        # 在该代码块中不进行梯度计算
        with torch.no_grad():
            # 使用 tokenizer 对观察数据进行编码，并得到 obs_tokens
            obs_tokens = tokenizer.encode(batch['observations'], should_preprocess=True).tokens  # (BL, K)

        # 将动作数据调整形状，将形状从 (b, l) 调整为 (b, l, 1)，这样可以与观察数据进行拼接
        act_tokens = rearrange(batch['actions'], 'b l -> b l 1')
        # obs_tokens act_tokens 在最后一个维度拼接，形成输入 tokens,
        tokens = rearrange(torch.cat((obs_tokens, act_tokens), dim=2), 'b l k1 -> b (l k1)')  # (B, L(K+1))

        # 将拼接后的 tokens 输入到模型中，得到模型的输出 outputs
        outputs = self(tokens)

        # 生成标签（损失的标签，包括观察的标签、奖励标签和结束标签。）
        labels_observations, labels_rewards, labels_ends = self.compute_labels_world_model(obs_tokens, batch['rewards'], batch['ends'], batch['mask_padding'])

        # B：批次大小。T：时间步数。O：每个时间步的输出维度
        # 将模型的观察输出调整形状，使得它与标签的形状匹配，便于计算损失
        # [:, :-1] 表示在时间维度上选取从第 0 到第 T-1 个时间步（不包括最后一个时间步）
        logits_observations = rearrange(outputs.logits_observations[:, :-1], 'b t o -> (b t) o')
        
        # 计算loss
        # 观察数据的交叉熵损失
        loss_obs = F.cross_entropy(logits_observations, labels_observations)
        # 奖励的交叉熵损失
        loss_rewards = F.cross_entropy(rearrange(outputs.logits_rewards, 'b t e -> (b t) e'), labels_rewards)
        # 结束标志的交叉熵损失
        loss_ends = F.cross_entropy(rearrange(outputs.logits_ends, 'b t e -> (b t) e'), labels_ends)

        return LossWithIntermediateLosses(loss_obs=loss_obs, loss_rewards=loss_rewards, loss_ends=loss_ends)

    def compute_labels_world_model(self, obs_tokens: torch.Tensor, rewards: torch.Tensor, ends: torch.Tensor, mask_padding: torch.BoolTensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 确保每个样本最多只能有一个结束信号。对于每个样本（沿 dim=1），ends 的和不能大于 1
        assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        # 通过对 mask_padding 取逻辑反，得到不需要被填充的位置。mask_fill 为 True 的位置代表有效数据，而为 False 的位置表示需要填充
        mask_fill = torch.logical_not(mask_padding)
        # 对调整形状后的张量，忽略第一个观测
        labels_observations = rearrange(obs_tokens.masked_fill(mask_fill.unsqueeze(-1).expand_as(obs_tokens), -100), 'b t k -> b (t k)')[:, 1:]
        # rewards.sign() + 1：将奖励的值映射到 {0, 1, 2} 的范围
        # 将数据类型转换为 long 类型，以便在计算交叉熵损失时使用
        labels_rewards = (rewards.sign() + 1).masked_fill(mask_fill, -100).long()  # Rewards clipped to {-1, 0, 1}
        # 对于 ends 张量中的无效位置，使用 -100 进行填充
        labels_ends = ends.masked_fill(mask_fill, -100)
        # 都展平为一维张量
        return labels_observations.reshape(-1), labels_rewards.reshape(-1), labels_ends.reshape(-1)
