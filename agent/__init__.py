"""
Pokemon RL Agent Package

包含:
- agent.py: PPO/DQN 算法实现
- reward_function.py: 可自定义的奖励函数
- environment/pokemon_env.py: Gym 风格的环境封装
- train.py: 训练脚本
"""

from .environment.pokemon_env import PokemonEnv
from .rl.agent import PPOAgent, DQNAgent, RandomAgent
from .rl.reward_function import (
    RewardFunction,
    CombinedReward,
    HPChangeReward,
    LevelUpReward,
    BattleWinReward,
    ExplorationReward,
    ItemReward,
    StepPenalty,
    ImageChangeReward,
    ScreenStaticPenalty,
    SurvivalReward,
    CustomRewardFunction,
    REWARD_CONFIGS,
)

__all__ = [
    'PPOAgent',
    'DQNAgent',
    'RandomAgent',
    'RewardFunction',
    'CombinedReward',
    'HPChangeReward',
    'LevelUpReward',
    'BattleWinReward',
    'ExplorationReward',
    'ItemReward',
    'StepPenalty',
    'ImageChangeReward',
    'ScreenStaticPenalty',
    'SurvivalReward',
    'CustomRewardFunction',
    'REWARD_CONFIGS',
    'PokemonEnv',
]

