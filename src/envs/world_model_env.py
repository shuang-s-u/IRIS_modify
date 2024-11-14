import random
from typing import List, Optional, Union

import gym
from einops import rearrange
import numpy as np
<<<<<<< HEAD
=======
# PIL 是 Python Imaging Library 的缩写，它是一个用于图像处理的第三方库。PIL 提供了一组方便的工具，用于图像的打开、操作和保存。
# 不过，PIL 已经被弃用，它的后继者是 Pillow，Pillow 是一个兼容 PIL 的库，提供了类似的接口，使用起来非常方便。
>>>>>>> remotecopy
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
import torchvision


<<<<<<< HEAD
class WorldModelEnv:

    def __init__(self, tokenizer: torch.nn.Module, world_model: torch.nn.Module, device: Union[str, torch.device], env: Optional[gym.Env] = None) -> None:

        self.device = torch.device(device)
=======
'''
tokenizer: 用于对观测进行编码和解码。
world_model: 预测环境状态（例如奖励、下一个状态等）。
device: 指定使用的设备（如 GPU 或 CPU）。
env: 可选的实际 Gym 环境，用于与真实环境进行交互。
Union 是 Python typing 模块中的一种类型提示，用于表示可以接受多个不同类型的变量
gym 是一个用于 构建和模拟强化学习环境 的库，由 OpenAI 开发。它提供了一系列标准化的模拟环境，帮助研究人员和开发者训练和测试强化学习算法。gym 中的环境封装了典型的强化学习问题，如经典控制（倒立摆、摆动等）、机器人控制、2D/3D 游戏等。
'''
class WorldModelEnv:

    def __init__(self, tokenizer: torch.nn.Module, world_model: torch.nn.Module, device: Union[str, torch.device], env: Optional[gym.Env] = None) -> None:
        # 将传入的 device 参数转换为 PyTorch 的设备对象，以便后续代码在该设备上运行
        self.device = torch.device(device)
        # to 移动到指定的设备上
        # 调用 .eval() 方法将这些模型设置为评估模式（即不进行 dropout 或 batchnorm 的训练），以保证一致的推理结果。
>>>>>>> remotecopy
        self.world_model = world_model.to(self.device).eval()
        self.tokenizer = tokenizer.to(self.device).eval()

        self.keys_values_wm, self.obs_tokens, self._num_observations_tokens = None, None, None

<<<<<<< HEAD
=======
        # 将传入的 gym 环境保存为类的属性。如果未提供，则值为 None
>>>>>>> remotecopy
        self.env = env

    @property
    def num_observations_tokens(self) -> int:
        return self._num_observations_tokens

    @torch.no_grad()
    def reset(self) -> torch.FloatTensor:
<<<<<<< HEAD
        assert self.env is not None
=======
        # 确保环境不为空
        assert self.env is not None
        # 调用 self.env.reset() 来重置环境，返回初始观察值（通常为一个图像）
        # 使用 torchvision.transforms.functional.to_tensor 将观察值转换为 PyTorch 张量
        # 将张量移动到指定的设备（如 GPU 或 CPU）
        # .unsqueeze(0)：增加一个新的批次维度，使得形状为 (1, C, H, W)，其中 C 是通道数，H 和 W 是高度和宽度
>>>>>>> remotecopy
        obs = torchvision.transforms.functional.to_tensor(self.env.reset()).to(self.device).unsqueeze(0)  # (1, C, H, W) in [0., 1.]
        return self.reset_from_initial_observations(obs)

    @torch.no_grad()
    def reset_from_initial_observations(self, observations: torch.FloatTensor) -> torch.FloatTensor:
<<<<<<< HEAD
        obs_tokens = self.tokenizer.encode(observations, should_preprocess=True).tokens    # (B, C, H, W) -> (B, K)
        _, num_observations_tokens = obs_tokens.shape
=======
        # 其中 B 是批次大小，K 是经过编码后的 token 数量。
        obs_tokens = self.tokenizer.encode(observations, should_preprocess=True).tokens    # (B, C, H, W) -> (B, K)
        # num_observations_tokens 表示每个观察值的 token 数量
        _, num_observations_tokens = obs_tokens.shape
        # 当检查 self.num_observations_tokens 是否为 None 时，其实是在检查私有变量 self._num_observations_tokens 是否已被赋值
>>>>>>> remotecopy
        if self.num_observations_tokens is None:
            self._num_observations_tokens = num_observations_tokens

        _ = self.refresh_keys_values_with_initial_obs_tokens(obs_tokens)
        self.obs_tokens = obs_tokens

        return self.decode_obs_tokens()

    @torch.no_grad()
<<<<<<< HEAD
    def refresh_keys_values_with_initial_obs_tokens(self, obs_tokens: torch.LongTensor) -> torch.FloatTensor:
        n, num_observations_tokens = obs_tokens.shape
        assert num_observations_tokens == self.num_observations_tokens
        self.keys_values_wm = self.world_model.transformer.generate_empty_keys_values(n=n, max_tokens=self.world_model.config.max_tokens)
=======
    # 利用初始观测的 token 刷新键和值（keys 和 values）
    def refresh_keys_values_with_initial_obs_tokens(self, obs_tokens: torch.LongTensor) -> torch.FloatTensor:
        # n batch
        n, num_observations_tokens = obs_tokens.shape
        assert num_observations_tokens == self.num_observations_tokens
        self.keys_values_wm = self.world_model.transformer.generate_empty_keys_values(n=n, max_tokens=self.world_model.config.max_tokens)
        # 模型的前向传播会基于这些 tokens 计算新的键和值
>>>>>>> remotecopy
        outputs_wm = self.world_model(obs_tokens, past_keys_values=self.keys_values_wm)
        return outputs_wm.output_sequence  # (B, K, E)

    @torch.no_grad()
<<<<<<< HEAD
    def step(self, action: Union[int, np.ndarray, torch.LongTensor], should_predict_next_obs: bool = True) -> None:
        assert self.keys_values_wm is not None and self.num_observations_tokens is not None

        num_passes = 1 + self.num_observations_tokens if should_predict_next_obs else 1

        output_sequence, obs_tokens = [], []

        if self.keys_values_wm.size + num_passes > self.world_model.config.max_tokens:
            _ = self.refresh_keys_values_with_initial_obs_tokens(self.obs_tokens)

        token = action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.long)
        token = token.reshape(-1, 1).to(self.device)  # (B, 1)

        for k in range(num_passes):  # assumption that there is only one action token.

            outputs_wm = self.world_model(token, past_keys_values=self.keys_values_wm)
            output_sequence.append(outputs_wm.output_sequence)

            if k == 0:
                reward = Categorical(logits=outputs_wm.logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1   # (B,)
=======
    # 模拟环境中的一步操作
    # 该方法的输入 action 可以是整数、Numpy 数组或 PyTorch 张量
    # 该方法接收一个动作 (action)，并基于当前的世界模型预测下一个观察 (observation)、奖励 (reward)、和是否结束 (done) 等信息。
    def step(self, action: Union[int, np.ndarray, torch.LongTensor], should_predict_next_obs: bool = True) -> None:
        # 确保在调用 step 方法之前，keys_values_wm 和 num_observations_tokens 已正确设置
        assert self.keys_values_wm is not None and self.num_observations_tokens is not None

        # 1 表示模型至少需要进行一次前向传播，用来生成与动作相关的输出（如奖励和结束标志）
        # self.num_observations_tokens 表示与观察有关的 token 数目。如果需要生成下一个观察，就需要在第一个前向传播之后，生成更多的 token 来构造新的观察。
        num_passes = 1 + self.num_observations_tokens if should_predict_next_obs else 1

        # 初始化输出序列和观察 tokens 的列表
        output_sequence, obs_tokens = [], []

        # 判断是否需要刷新缓存
        if self.keys_values_wm.size + num_passes > self.world_model.config.max_tokens:
            _ = self.refresh_keys_values_with_initial_obs_tokens(self.obs_tokens)

        # 格式化动作 token
        # step 函数是在推理（inference）过程中使用的，不需要计算反向传播的梯度。
        token = action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.long)
        # 每个样本包含一个 token
        token = token.reshape(-1, 1).to(self.device)  # (B, 1)

        # 通过给定的动作 action 在环境中执行一步的操作，并获取模型预测的奖励、是否结束的标志以及下一个状态
        for k in range(num_passes):  # assumption that there is only one action token.
            
            # 将当前的 token（动作）和之前的键值对缓存（self.keys_values_wm）作为输入
            outputs_wm = self.world_model(token, past_keys_values=self.keys_values_wm)
            output_sequence.append(outputs_wm.output_sequence)
            
            # 只在第一次循环（即动作token执行后）计算奖励和结束标志
            if k == 0:
                # Categorical 是 PyTorch 中用于表示类别分布的类
                # logits 是原始的未归一化的分数，可以通过 softmax 转化为概率。
                # - 1：将采样的结果减去 1，以使奖励值在范围 {-1, 0, 1} 内
                reward = Categorical(logits=outputs_wm.logits_rewards).sample().float().cpu().numpy().reshape(-1) - 1   # (B,)
                # astype(bool)：将采样得到的整数值转换为布尔值，表示是否结束 (True 表示结束，False 表示未结束)。
>>>>>>> remotecopy
                done = Categorical(logits=outputs_wm.logits_ends).sample().cpu().numpy().astype(bool).reshape(-1)       # (B,)

            if k < self.num_observations_tokens:
                token = Categorical(logits=outputs_wm.logits_observations).sample()
                obs_tokens.append(token)

        output_sequence = torch.cat(output_sequence, dim=1)   # (B, 1 + K, E)
        self.obs_tokens = torch.cat(obs_tokens, dim=1)        # (B, K)

        obs = self.decode_obs_tokens() if should_predict_next_obs else None
        return obs, reward, done, None

    @torch.no_grad()
<<<<<<< HEAD
    def render_batch(self) -> List[Image.Image]:
        frames = self.decode_obs_tokens().detach().cpu()
        frames = rearrange(frames, 'b c h w -> b h w c').mul(255).numpy().astype(np.uint8)
        return [Image.fromarray(frame) for frame in frames]

    @torch.no_grad()
    def decode_obs_tokens(self) -> List[Image.Image]:
        embedded_tokens = self.tokenizer.embedding(self.obs_tokens)     # (B, K, E)
        z = rearrange(embedded_tokens, 'b (h w) e -> b e h w', h=int(np.sqrt(self.num_observations_tokens)))
        rec = self.tokenizer.decode(z, should_postprocess=True)         # (B, C, H, W)
        return torch.clamp(rec, 0, 1)

    @torch.no_grad()
    def render(self):
=======
    # 生成并返回一个包含一批观测的图像列表。
    def render_batch(self) -> List[Image.Image]:
        # 使用 detach() 方法使其与计算图分离，随后将张量移到 CPU（通过 .cpu()），以便进一步处理。
        frames = self.decode_obs_tokens().detach().cpu()
        # b h w c 更符合图像处理工具（如 PIL）使用的标准格式 (height, width, channels)
        # 将张量中的浮点值从 [0, 1] 映射到 [0, 255]
        # 将张量转换为 NumPy 数组，并将数据类型转换为 np.uint8，这是图像数据通常的格式
        frames = rearrange(frames, 'b c h w -> b h w c').mul(255).numpy().astype(np.uint8)
        # Image.fromarray 方法将 NumPy 数组转换为图像对象，并将所有的图像存入列表中返回
        return [Image.fromarray(frame) for frame in frames]

    @torch.no_grad()
    # 解码观察 tokens（obs_tokens），将其转换为图像的形式
    def decode_obs_tokens(self) -> List[Image.Image]:
        # 使用 tokenizer 中的嵌入层，将观测 tokens (self.obs_tokens) 转换为对应的嵌入表示。
        embedded_tokens = self.tokenizer.embedding(self.obs_tokens)     # （B, K）-> (B, K, E)
        # h 被设定为 int(np.sqrt(self.num_observations_tokens))，假设 tokens 的数量是一个完全平方数，表示将 tokens 重新排列为一个正方形
        z = rearrange(embedded_tokens, 'b (h w) e -> b e h w', h=int(np.sqrt(self.num_observations_tokens)))
        # 将嵌入后的特征图解码回图像
        rec = self.tokenizer.decode(z, should_postprocess=True)       # (B, C, H, W)
        # 将解码后的图像的值限制在 [0, 1] 范围内，以确保像素值不超出这个区间。
        return torch.clamp(rec, 0, 1)

    @torch.no_grad()
    # 并返回其中的第一张图像
    def render(self):
        # 检查当前的 obs_tokens 的形状是否符合预期。这里假定批次大小为 1，即只渲染单个观测值
>>>>>>> remotecopy
        assert self.obs_tokens.shape == (1, self.num_observations_tokens)
        return self.render_batch()[0]
