"""
PPO (Proximal Policy Optimization) Agent for Pokemon

基于 Stable-Baselines3 的 PPO 实现
"""

import os
import random
import sys
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import Adam

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class RolloutBuffer:
    """存储 rollout 数据"""
    observations: List[Dict[str, np.ndarray]]
    actions: List[np.ndarray]
    rewards: List[np.ndarray]
    dones: List[np.ndarray]
    log_probs: List[np.ndarray]
    values: List[np.ndarray]


class CNNPolicy(nn.Module):
    """CNN 策略网络 - 处理图像和状态"""

    def __init__(self, image_shape: Tuple, state_dim: int, n_actions: int):
        super().__init__()

        # 图像特征提取
        h, w, c = image_shape

        # 简单的 CNN
        self.image_net = nn.Sequential(
            nn.Conv2d(c, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # 计算 CNN 输出大小
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            cnn_out = self.image_net(dummy).shape[1]

        # 状态特征提取
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
        )

        # 策略头
        self.actor = nn.Sequential(
            nn.Linear(cnn_out + 64, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

        # 价值头
        self.critic = nn.Sequential(
            nn.Linear(cnn_out + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, image, state):
        # 提取特征
        img_feat = self.image_net(image)
        state_feat = self.state_net(state)

        # 合并特征
        combined = torch.cat([img_feat, state_feat], dim=-1)

        # 策略和价值
        logits = self.actor(combined)
        value = self.critic(combined)

        return logits, value


class PPOAgent:
    """
    PPO Agent

    使用 Proximal Policy Optimization 算法
    """

    def __init__(
        self,
        image_shape: Tuple[int, int, int] = (84, 84, 3),
        state_dim: int = 50,
        n_actions: int = 22,
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        k_epochs: int = 4,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        初始化 PPO Agent

        Args:
            image_shape: 图像形状 (H, W, C)
            state_dim: 状态向量维度
            n_actions: 动作数量
            lr: 学习率
            gamma: 折扣因子
            eps_clip: PPO 裁剪范围
            k_epochs: 每次更新重复的 epoch 数
            ent_coef: 熵系数
            vf_coef: 价值损失系数
            max_grad_norm: 梯度裁剪
            device: 设备
        """
        self.image_shape = image_shape
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.device = device

        # 网络
        self.policy = CNNPolicy(image_shape, state_dim, n_actions).to(device)
        self.old_policy = CNNPolicy(image_shape, state_dim, n_actions).to(device)
        self.old_policy.load_state_dict(self.policy.state_dict())

        # 优化器
        self.optimizer = Adam(self.policy.parameters(), lr=lr)

        # Buffer
        self.buffer = RolloutBuffer([], [], [], [], [], [])

        # 训练统计
        self.training_stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'reward': [],
        }

    def select_action(self, observation: Dict[str, np.ndarray], training: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        选择动作 (支持批量)

        Args:
            observation: 观察 {'image': (N, H, W, C), 'state': (N, D)}
            training: 是否在训练模式

        Returns:
            action: 选择的动作 (N,)
            log_prob: 动作的对数概率 (N,)
            value: 状态价值 (N,)
        """
        with torch.no_grad():
            # 转换观察为 tensor
            # image: (N, H, W, C) -> (N, C, H, W)
            image = torch.FloatTensor(observation['image']).permute(0, 3, 1, 2).to(self.device)
            state = torch.FloatTensor(observation['state']).to(self.device)

            # 如果输入是单个样本 (H, W, C)，增加 batch 维度
            if image.dim() == 3:
                image = image.unsqueeze(0)
                state = state.unsqueeze(0)

            # 获取策略分布
            logits, value = self.policy(image, state)
            probs = torch.softmax(logits, dim=-1)

            if training:
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            else:
                action = probs.argmax(dim=-1)
                log_prob = torch.log(probs.gather(-1, action.unsqueeze(-1)) + 1e-8).squeeze(-1)

            return action.cpu().numpy(), log_prob.cpu().numpy(), value.squeeze(-1).cpu().numpy()

    def store_transition(self, obs, action, reward, done, log_prob, value):
        """存储 transition"""
        self.buffer.observations.append(obs)
        self.buffer.actions.append(action)
        self.buffer.rewards.append(reward)
        self.buffer.dones.append(done)
        self.buffer.log_probs.append(log_prob)
        self.buffer.values.append(value)

    def compute_returns(self):
        """计算 returns 和 advantages (支持批量)"""
        rewards = self.buffer.rewards
        dones = self.buffer.dones
        values = self.buffer.values

        # 获取环境数量 (假设所有 step 的 batch size 一致)
        n_envs = len(rewards[0]) if isinstance(rewards[0], np.ndarray) else 1

        returns = []
        discounted_reward = np.zeros(n_envs)
        advantages = []

        for reward, done, value in zip(reversed(rewards), reversed(dones), reversed(values)):
            # 如果 done 为 True (1)，则重置 discounted_reward
            # R_t = r_t + gamma * R_{t+1} * (1 - d_t)

            # 确保 done 是 numpy array
            if not isinstance(done, np.ndarray):
                done = np.array([done])

            discounted_reward = reward + self.gamma * discounted_reward * (1 - done)
            advantage = discounted_reward - value

            returns.insert(0, discounted_reward)
            advantages.insert(0, advantage)

        return returns, advantages

    def update(self):
        """更新策略"""
        if len(self.buffer.rewards) == 0:
            return

        # 计算 returns 和 advantages
        returns, advantages = self.compute_returns()

        # 展平数据 (N_steps, N_envs, ...) -> (N_steps * N_envs, ...)

        # Observations
        # obs['image']: list of (N_envs, H, W, C) -> (N_steps * N_envs, C, H, W)
        obs_images = np.concatenate([o['image'] for o in self.buffer.observations])
        obs_states = np.concatenate([o['state'] for o in self.buffer.observations])

        images = torch.FloatTensor(obs_images).permute(0, 3, 1, 2).to(self.device)
        states = torch.FloatTensor(obs_states).to(self.device)

        # Actions, LogProbs, Values, Returns, Advantages
        actions = torch.LongTensor(np.concatenate(self.buffer.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.concatenate(self.buffer.log_probs)).to(self.device)
        old_values = torch.FloatTensor(np.concatenate(self.buffer.values)).to(self.device)
        returns = torch.FloatTensor(np.concatenate(returns)).to(self.device)
        advantages = torch.FloatTensor(np.concatenate(advantages)).to(self.device)

        # 归一化 advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 多次 epoch 更新
        for _ in range(self.k_epochs):
            # 获取新的策略分布
            logits, values = self.policy(images, states)
            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)

            # 计算策略损失
            new_log_probs = dist.log_prob(actions)
            ratio = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # 计算价值损失
            value_loss = nn.functional.mse_loss(values.squeeze(-1), returns)

            # 计算熵
            entropy = dist.entropy().mean()

            # 总损失
            loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

            # 更新
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # 记录统计
            self.training_stats['policy_loss'].append(policy_loss.item())
            self.training_stats['value_loss'].append(value_loss.item())
            self.training_stats['entropy'].append(entropy.item())

        # 清空 buffer
        self.buffer = RolloutBuffer([], [], [], [], [], [])

        # 复制新策略到旧策略
        self.old_policy.load_state_dict(self.policy.state_dict())

    def save(self, path: str):
        """保存模型"""
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.old_policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def get_stats(self) -> Dict:
        """获取训练统计"""
        stats = {}
        for key, values in self.training_stats.items():
            if values:
                stats[key] = np.mean(values[-100:])
        return stats


class RandomAgent:
    """随机 Agent（用于基准测试）"""

    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def select_action(self, observation, training=False):
        # Handle batch
        if isinstance(observation['image'], np.ndarray) and observation['image'].ndim == 4:
            batch_size = observation['image'].shape[0]
            return np.random.randint(0, self.n_actions, size=batch_size), np.zeros(batch_size), np.zeros(batch_size)

        action = random.randint(0, self.n_actions - 1)
        return action, 0.0, 0.0

    def store_transition(self, *args):
        pass

    def update(self):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent

    简单的 Deep Q-Network 实现
    """

    def __init__(
        self,
        image_shape: Tuple = (84, 84, 3),
        state_dim: int = 50,
        n_actions: int = 22,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.1,
        buffer_size: int = 100000,
        batch_size: int = 32,
        target_update_freq: int = 1000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        self.steps = 0

        # Q 网络
        self.q_network = CNNPolicy(image_shape, state_dim, n_actions).to(device)
        self.target_network = CNNPolicy(image_shape, state_dim, n_actions).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, observation, training=True):
        # TODO: Support batch for DQN
        if training and random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1), 0.0, 0.0

        with torch.no_grad():
            image = torch.FloatTensor(observation['image']).permute(2, 0, 1).unsqueeze(0).to(self.device)
            state = torch.FloatTensor(observation['state']).unsqueeze(0).to(self.device)
            q_values, _ = self.q_network(image, state)
            action = q_values.argmax(dim=-1).item()
            return action, 0.0, q_values[0, action].item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.steps += 1

        # 更新 epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # 采样
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # 转换为 tensor
        images = torch.FloatTensor([s['image'].transpose(2, 0, 1) for s in states]).to(self.device)
        image_next = torch.FloatTensor([s['image'].transpose(2, 0, 1) for s in next_states]).to(self.device)
        state_batch = torch.FloatTensor([s['state'] for s in states]).to(self.device)
        state_next = torch.FloatTensor([s['state'] for s in next_states]).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # 计算 Q 值
        q_values, _ = self.q_network(images, state_batch)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算目标 Q 值
        with torch.no_grad():
            next_q_values, _ = self.target_network(image_next, state_next)
            next_q_value = next_q_values.max(1)[0]
            target_q_value = rewards + (1 - dones.float()) * self.gamma * next_q_value

        # 计算损失
        loss = nn.functional.mse_loss(q_value, target_q_value)

        # 更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标网络
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {'loss': loss.item(), 'epsilon': self.epsilon}

    def save(self, path: str):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']

