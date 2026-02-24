"""
Gym-style Environment for Pokemon Emerald

将 emulator.py 封装为 Gym 风格的环境
"""

import os
import sys
from typing import Dict, Tuple, Optional

import gym
import numpy as np
from gym import spaces

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.environment.emulator import Emulator


class PokemonEnv(gym.Env):
    """
    Pokemon Emerald 的 Gym 环境

    动作空间: 离散空间，包含所有可能的按键组合
    观察空间: 图像 + 游戏状态信息
    """

    # 定义可用的动作
    ACTIONS = [
        # 方向
        'A', 'B',  # 按钮
        'START', 'SELECT',
        'UP', 'DOWN', 'LEFT', 'RIGHT',  # 方向键
        # 组合动作
        'UP+A', 'DOWN+A', 'LEFT+A', 'RIGHT+A',
        'UP+B', 'DOWN+B', 'LEFT+B', 'RIGHT+B',
        # 特殊动作
        'NOOP',  # 不操作
    ]

    # 按钮到动作的映射
    BUTTON_MAP = {
        'A': 0,
        'B': 1,
        'START': 2,
        'SELECT': 3,
        'UP': 4,
        'DOWN': 5,
        'LEFT': 6,
        'RIGHT': 7,
    }

    def __init__(
        self,
        rom_path: str = "pokeemerald.gba",
        frame_skip: int = 10,
        image_size: Tuple[int, int] = (84, 84),
        use_state_dict: bool = True,
    ):
        """
        初始化环境

        Args:
            rom_path: ROM 文件路径
            frame_skip: 每次动作跳过的帧数
            image_size: 图像缩放大小
            use_state_dict: 是否使用状态字典（包含游戏详细信息）
        """
        super().__init__()

        self.rom_path = rom_path
        self.frame_skip = frame_skip
        self.image_size = image_size
        self.use_state_dict = use_state_dict

        # 加载游戏rom获取正确路径
        if not os.path.exists(rom_path):
            # 尝试在当前目录查找
            base_dir = os.path.dirname(os.path.abspath(__file__))
            project_dir = os.path.dirname(base_dir)
            test_path = os.path.join(project_dir, rom_path)
            if os.path.exists(test_path):
                self.rom_path = test_path

        # 定义动作空间
        self.action_space = spaces.Discrete(len(self.ACTIONS))

        # 定义观察空间
        # 图像 + 游戏状态
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=255,
                shape=(*image_size, 3),
                dtype=np.uint8
            ),
            'state': spaces.Box(
                low=-999999, high=999999,
                shape=(50,),  # 状态向量大小
                dtype=np.float32
            )
        })

        # 状态记录
        self.emulator: Optional[Emulator] = None
        self.current_state: Optional[Dict] = None
        self.prev_state: Optional[Dict] = None
        self.step_count: int = 0

        # 奖励函数（由外部设置）
        self.reward_fn = None

    def _init_emulator(self):
        """初始化模拟器"""
        if self.emulator is None:
            self.emulator = Emulator(self.rom_path)
            if not self.emulator.core:
                raise RuntimeError("Failed to initialize emulator")

    def reset(self) -> Dict[str, np.ndarray]:
        """重置环境"""
        self._init_emulator()
        self.step_count = 0

        # 重置模拟器
        self.emulator.core.reset()

        # 运行几帧让游戏初始化
        for _ in range(30):
            self.emulator.run_frame()

        # 获取初始状态
        self.prev_state = self.emulator.get_state(include_screenshot=True)
        self.current_state = self.prev_state

        # 重置奖励函数状态
        if self.reward_fn:
            self.reward_fn.reset()

        return self._get_observation()

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict]:
        """
        执行动作

        Args:
            action: 动作索引

        Returns:
            observation: 观察
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        # 将动作转换为按键
        self._execute_action(action)

        # 运行多帧
        for _ in range(self.frame_skip):
            self.emulator.run_frame()

        # 获取新状态
        self.prev_state = self.current_state
        self.current_state = self.emulator.get_state(include_screenshot=True)

        self.step_count += 1

        # 计算奖励
        if self.reward_fn:
            reward = self.reward_fn.compute(self.prev_state, self.current_state, action)
        else:
            reward = 0.0

        # 检查是否结束
        done = self._is_done()

        # 额外信息
        info = {
            'step': self.step_count,
            'location': self.current_state.get('location', {}),
            'party': self.current_state.get('party', []),
            'bag': self.current_state.get('bag', {}),
            'screenshot': self.current_state.get('screenshot'),
        }

        return self._get_observation(), reward, done, info

    def _execute_action(self, action: int):
        """执行动作对应的按键"""
        action_name = self.ACTIONS[action]

        # 清除所有按键
        self.emulator.core.clear_keys(*self.BUTTON_MAP.values())

        if action_name == 'NOOP':
            return

        # 处理组合动作
        if '+' in action_name:
            parts = action_name.split('+')
            for part in parts:
                self._press_button(part.strip())
        else:
            self._press_button(action_name)

    def _press_button(self, button: str):
        """按下按钮"""
        if button in self.BUTTON_MAP:
            key = self.BUTTON_MAP[button]
            self.emulator.core.set_keys(key)

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """获取当前观察"""
        # 获取图像
        screenshot = self.current_state.get('screenshot')
        if screenshot is None:
            image = np.zeros((*self.image_size, 3), dtype=np.uint8)
        else:
            # 调整图像大小
            from PIL import Image
            pil_img = screenshot.resize(self.image_size, Image.LANCZOS)
            image = np.array(pil_img, dtype=np.uint8)

        # 获取状态向量
        state_vector = self._get_state_vector()

        return {
            'image': image,
            'state': state_vector
        }

    def _get_state_vector(self) -> np.ndarray:
        """将游戏状态转换为向量"""
        state = self.current_state

        # 初始化向量
        vec = np.zeros(50, dtype=np.float32)

        # 位置信息 (0-5)
        loc = state.get('location', {})
        vec[0] = loc.get('x', 0)
        vec[1] = loc.get('y', 0)
        vec[2] = loc.get('map_group', 0)
        vec[3] = loc.get('map_num', 0)

        # 队伍信息 (4-23)
        party = state.get('party', [])
        for i, mon in enumerate(party[:5]):  # 最多5只
            idx = 4 + i * 4
            vec[idx] = mon.get('level', 0)
            vec[idx + 1] = mon.get('hp', 0)
            vec[idx + 2] = mon.get('max_hp', 0)
            vec[idx + 3] = mon.get('species_id', 0)

        # 包裹物品数量 (24-28)
        bag = state.get('bag', {})
        pocket_idx = 24
        for pocket_name, items in bag.items():
            if pocket_idx >= 28:
                break
            vec[pocket_idx] = len(items)
            pocket_idx += 1

        return vec

    def _is_done(self) -> bool:
        """检查是否结束"""
        # 检查是否团灭
        party = self.current_state.get('party', [])
        if not party:
            # 如果队伍为空（例如游戏刚开始），则不算团灭
            all_dead = False
        else:
            all_dead = all(mon.get('hp', 0) == 0 for mon in party)

        # 或者达到最大步数
        max_steps = 100000
        if self.step_count >= max_steps:
            return True

        return all_dead

    def render(self, mode='human'):
        """渲染（仅在 human 模式下有效）"""
        if mode == 'human':
            screenshot = self.current_state.get('screenshot')
            if screenshot:
                from PIL import Image
                import cv2

                # 调整大小用于显示
                display = screenshot.resize((480, 320), Image.LANCZOS)
                return np.array(display)
        return None

    def close(self):
        """关闭环境"""
        if self.emulator:
            # 清理资源
            pass

    def set_reward_function(self, reward_fn):
        """设置奖励函数"""
        self.reward_fn = reward_fn

    def get_reward_state(self) -> Dict:
        """获取奖励函数状态"""
        if self.reward_fn:
            return self.reward_fn.state_dict()
        return {}

    def set_reward_state(self, state_dict: Dict):
        """设置奖励函数状态"""
        if self.reward_fn:
            self.reward_fn.load_state_dict(state_dict)

    def get_state_dict(self) -> Dict:
        """获取完整的状态字典"""
        return self.current_state


# 注册环境
gym.envs.register(
    id='PokemonEmerald-v0',
    entry_point='environment.pokemon_env:PokemonEnv',
)

