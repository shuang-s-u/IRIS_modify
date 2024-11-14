from dataclasses import dataclass
from typing import Any, Optional, Union
import sys

from einops import rearrange
import numpy as np
import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from dataset import Batch
from envs.world_model_env import WorldModelEnv
from models.tokenizer import Tokenizer
from models.world_model import WorldModel
from utils import compute_lambda_returns, LossWithIntermediateLosses


@dataclass
class ActorCriticOutput:
<<<<<<< HEAD
    logits_actions: torch.FloatTensor
=======
    # 表示动作的原始输出分数（logits）
    logits_actions: torch.FloatTensor
    # 表示状态值函数的均值。这是用来估计当前状态的“好坏”的数值
>>>>>>> remotecopy
    means_values: torch.FloatTensor


@dataclass
class ImagineOutput:
<<<<<<< HEAD
    observations: torch.ByteTensor
    actions: torch.LongTensor
    logits_actions: torch.FloatTensor
    values: torch.FloatTensor
    rewards: torch.FloatTensor
    ends: torch.BoolTensor


=======
    # 观察值的张量，使用 ByteTensor，通常是表示离散化或者图像数据的较小数值
    # ByteTensor 的数据范围是 0-255
    observations: torch.ByteTensor
    # 在模拟环境中采取的动作，是一个 LongTensor，因为动作通常是离散的整型值
    actions: torch.LongTensor
    # 表示为动作输出的 logits 值，通常可以使用 softmax 来生成相应的动作概率分布
    logits_actions: torch.FloatTensor
    # 从 Critic 模块中估计的状态值，用于衡量每个状态的价值
    values: torch.FloatTensor
    # 在模拟环境中采取动作后获得的奖励值
    rewards: torch.FloatTensor
    # 表示当前回合是否已经结束
    ends: torch.BoolTensor


'''
act_vocab_size：动作的词汇表大小，表示智能体可以采取的所有不同动作的数量
use_original_obs：布尔变量，用于决定是否使用原始观察值
'''
>>>>>>> remotecopy
class ActorCritic(nn.Module):
    def __init__(self, act_vocab_size, use_original_obs: bool = False) -> None:
        super().__init__()
        self.use_original_obs = use_original_obs
<<<<<<< HEAD
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
=======
        # 这些卷积层用于提取输入观察（例如环境中的图像）的特征。
        '''
        第一层卷积层将输入通道从 3 增加到 32，然后使用最大池化层将特征图的尺寸减半。
        第二层卷积层继续提取更高层次的特征，同样输出 32 个通道，然后再进行池化。
        第三层卷积层将输出通道增加到 64，并进行池化。
        第四层卷积层保持通道数为 64，并继续进行池化。
        这些卷积和池化层帮助模型从输入观察中逐渐提取有用的特征，并对图像的空间维度进行下采样。
        '''
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        # 池化窗口的大小为 2 x 2, 这步之后会使特征图的空间尺寸缩小一半。
>>>>>>> remotecopy
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        self.lstm_dim = 512
<<<<<<< HEAD
=======
        # LSTM（长短期记忆网络）基本单元
>>>>>>> remotecopy
        self.lstm = nn.LSTMCell(1024, self.lstm_dim)
        self.hx, self.cx = None, None

        self.critic_linear = nn.Linear(512, 1)
<<<<<<< HEAD
=======
        # 
>>>>>>> remotecopy
        self.actor_linear = nn.Linear(512, act_vocab_size)

    def __repr__(self) -> str:
        return "actor_critic"

    def clear(self) -> None:
        self.hx, self.cx = None, None

    def reset(self, n: int, burnin_observations: Optional[torch.Tensor] = None, mask_padding: Optional[torch.Tensor] = None) -> None:
        device = self.conv1.weight.device
        self.hx = torch.zeros(n, self.lstm_dim, device=device)
        self.cx = torch.zeros(n, self.lstm_dim, device=device)
        if burnin_observations is not None:
            assert burnin_observations.ndim == 5 and burnin_observations.size(0) == n and mask_padding is not None and burnin_observations.shape[:2] == mask_padding.shape
            for i in range(burnin_observations.size(1)):
                if mask_padding[:, i].any():
                    with torch.no_grad():
                        self(burnin_observations[:, i], mask_padding[:, i])

    def prune(self, mask: np.ndarray) -> None:
        self.hx = self.hx[mask]
        self.cx = self.cx[mask]

    def forward(self, inputs: torch.FloatTensor, mask_padding: Optional[torch.BoolTensor] = None) -> ActorCriticOutput:
        assert inputs.ndim == 4 and inputs.shape[1:] == (3, 64, 64)
        assert 0 <= inputs.min() <= 1 and 0 <= inputs.max() <= 1
        assert mask_padding is None or (mask_padding.ndim == 1 and mask_padding.size(0) == inputs.size(0) and mask_padding.any())
        x = inputs[mask_padding] if mask_padding is not None else inputs

        x = x.mul(2).sub(1)
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        x = torch.flatten(x, start_dim=1)

        if mask_padding is None:
            self.hx, self.cx = self.lstm(x, (self.hx, self.cx))
        else:
            self.hx[mask_padding], self.cx[mask_padding] = self.lstm(x, (self.hx[mask_padding], self.cx[mask_padding]))

        logits_actions = rearrange(self.actor_linear(self.hx), 'b a -> b 1 a')
        means_values = rearrange(self.critic_linear(self.hx), 'b 1 -> b 1 1')

        return ActorCriticOutput(logits_actions, means_values)

    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, world_model: WorldModel, imagine_horizon: int, gamma: float, lambda_: float, entropy_weight: float, **kwargs: Any) -> LossWithIntermediateLosses:
        assert not self.use_original_obs
        outputs = self.imagine(batch, tokenizer, world_model, horizon=imagine_horizon)

        with torch.no_grad():
            lambda_returns = compute_lambda_returns(
                rewards=outputs.rewards,
                values=outputs.values,
                ends=outputs.ends,
                gamma=gamma,
                lambda_=lambda_,
            )[:, :-1]

        values = outputs.values[:, :-1]

        d = Categorical(logits=outputs.logits_actions[:, :-1])
        log_probs = d.log_prob(outputs.actions[:, :-1])
        loss_actions = -1 * (log_probs * (lambda_returns - values.detach())).mean()
        loss_entropy = - entropy_weight * d.entropy().mean()
        loss_values = F.mse_loss(values, lambda_returns)

        return LossWithIntermediateLosses(loss_actions=loss_actions, loss_values=loss_values, loss_entropy=loss_entropy)

    def imagine(self, batch: Batch, tokenizer: Tokenizer, world_model: WorldModel, horizon: int, show_pbar: bool = False) -> ImagineOutput:
        assert not self.use_original_obs
        initial_observations = batch['observations']
        mask_padding = batch['mask_padding']
        assert initial_observations.ndim == 5 and initial_observations.shape[2:] == (3, 64, 64)
        assert mask_padding[:, -1].all()
        device = initial_observations.device
        wm_env = WorldModelEnv(tokenizer, world_model, device)

        all_actions = []
        all_logits_actions = []
        all_values = []
        all_rewards = []
        all_ends = []
        all_observations = []

<<<<<<< HEAD
=======
        # 在这里设置断点
        import pdb; pdb.set_trace()
>>>>>>> remotecopy
        burnin_observations = torch.clamp(tokenizer.encode_decode(initial_observations[:, :-1], should_preprocess=True, should_postprocess=True), 0, 1) if initial_observations.size(1) > 1 else None
        self.reset(n=initial_observations.size(0), burnin_observations=burnin_observations, mask_padding=mask_padding[:, :-1])

        obs = wm_env.reset_from_initial_observations(initial_observations[:, -1])
        for k in tqdm(range(horizon), disable=not show_pbar, desc='Imagination', file=sys.stdout):

            all_observations.append(obs)

            outputs_ac = self(obs)
            action_token = Categorical(logits=outputs_ac.logits_actions).sample()
            obs, reward, done, _ = wm_env.step(action_token, should_predict_next_obs=(k < horizon - 1))

            all_actions.append(action_token)
            all_logits_actions.append(outputs_ac.logits_actions)
            all_values.append(outputs_ac.means_values)
            all_rewards.append(torch.tensor(reward).reshape(-1, 1))
            all_ends.append(torch.tensor(done).reshape(-1, 1))

        self.clear()

        return ImagineOutput(
            observations=torch.stack(all_observations, dim=1).mul(255).byte(),      # (B, T, C, H, W) in [0, 255]
            actions=torch.cat(all_actions, dim=1),                                  # (B, T)
            logits_actions=torch.cat(all_logits_actions, dim=1),                    # (B, T, #actions)
            values=rearrange(torch.cat(all_values, dim=1), 'b t 1 -> b t'),         # (B, T)
            rewards=torch.cat(all_rewards, dim=1).to(device),                       # (B, T)
            ends=torch.cat(all_ends, dim=1).to(device),                             # (B, T)
        )
