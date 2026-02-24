"""
Reward Function Module for Pokemon RL Agent

这个模块封装了各种奖励函数，你可以根据需要自定义和组合。
"""

from typing import Dict, List, Callable, Any

import numpy as np


class RewardFunction:
    """基础奖励函数类"""

    def __init__(self, name: str = "base"):
        self.name = name

    def compute(self, prev_state: Dict, current_state: Dict, action: int) -> float:
        """
        计算奖励

        Args:
            prev_state: 执行动作前的状态
            current_state: 执行动作后的状态
            action: 执行的动作

        Returns:
            奖励值
        """
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        """返回状态字典"""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载状态字典"""
        pass

    def reset(self):
        """重置奖励函数状态 (在 episode 开始时调用)"""
        pass


class CombinedReward(RewardFunction):
    """组合多个奖励函数"""

    def __init__(self, rewards: Dict[RewardFunction, float]):
        super().__init__("combined")
        self.rewards = rewards

    def compute(self, prev_state: Dict, current_state: Dict, action: int) -> float:
        total = 0.0
        for reward_fn, weight in self.rewards.items():
            total += weight * reward_fn.compute(prev_state, current_state, action)
        return total

    def state_dict(self) -> Dict[str, Any]:
        state = {}
        for i, (reward_fn, _) in enumerate(self.rewards.items()):
            state[f"{reward_fn.name}_{i}"] = reward_fn.state_dict()
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        for i, (reward_fn, _) in enumerate(self.rewards.items()):
            key = f"{reward_fn.name}_{i}"
            if key in state_dict:
                reward_fn.load_state_dict(state_dict[key])

    def reset(self):
        for reward_fn in self.rewards.keys():
            reward_fn.reset()


class HPChangeReward(RewardFunction):
    """基于宝可梦 HP 变化的奖励"""

    def __init__(self, hp_weight: float = 1.0):
        super().__init__("hp_change")
        self.hp_weight = hp_weight

    def compute(self, prev_state: Dict, current_state: Dict, action: int) -> float:
        prev_hp = self._get_total_hp(prev_state)
        curr_hp = self._get_total_hp(current_state)
        return (curr_hp - prev_hp) * self.hp_weight

    def _get_total_hp(self, state: Dict) -> float:
        party = state.get("party", [])
        if not party:
            return 0.0
        total = 0
        for mon in party:
            total += mon.get("hp", 0)
        return total


class LevelUpReward(RewardFunction):
    """升级奖励 - 基于宝可梦等级提升"""

    def __init__(self, level_weight: float = 10.0):
        super().__init__("level_up")
        self.level_weight = level_weight

    def compute(self, prev_state: Dict, current_state: Dict, action: int) -> float:
        prev_levels = self._get_total_levels(prev_state)
        curr_levels = self._get_total_levels(current_state)
        level_diff = sum(curr_levels) - sum(prev_levels)
        return level_diff * self.level_weight

    def _get_total_levels(self, state: Dict) -> List[int]:
        party = state.get("party", [])
        return [mon.get("level", 0) for mon in party]


class BattleWinReward(RewardFunction):
    """战斗胜利奖励"""

    def __init__(self, win_reward: float = 100.0):
        super().__init__("battle_win")
        self.win_reward = win_reward
        self.in_battle = False
        self.was_in_battle = False

    def compute(self, prev_state: Dict, current_state: Dict, action: int) -> float:
        # 检测是否刚结束战斗
        prev_battle = self._is_in_battle(prev_state)
        curr_battle = self._is_in_battle(current_state)

        if prev_battle and not curr_battle:
            # 战斗刚结束，检查是否胜利
            # 这里需要根据游戏状态判断胜负
            return self.win_reward
        return 0.0

    def _is_in_battle(self, state: Dict) -> bool:
        # 可以通过检测游戏状态判断是否在战斗中
        return state.get("in_battle", False)


class ExplorationReward(RewardFunction):
    """探索奖励 - 基于访问新地图位置"""

    def __init__(self, location_weight: float = 1.0, coord_weight: float = 0.05):
        super().__init__("exploration")
        self.location_weight = location_weight
        self.coord_weight = coord_weight
        self.visited_maps = set()
        self.visited_coords = set()

    def compute(self, prev_state: Dict, current_state: Dict, action: int) -> float:
        map_key = self._get_map_key(current_state)
        coord_key = self._get_coord_key(current_state)

        reward = 0.0

        # 检查是否是新地图
        if map_key not in self.visited_maps:
            self.visited_maps.add(map_key)
            reward += self.location_weight

        # 检查是否是新坐标
        if coord_key not in self.visited_coords:
            self.visited_coords.add(coord_key)
            reward += self.coord_weight

        return reward

    def _get_map_key(self, state: Dict) -> tuple:
        loc = state.get("location", {})
        return (loc.get("map_group", 0), loc.get("map_num", 0))

    def _get_coord_key(self, state: Dict) -> tuple:
        loc = state.get("location", {})
        return (loc.get("map_group", 0), loc.get("map_num", 0), loc.get("x", 0), loc.get("y", 0))

    def state_dict(self) -> Dict[str, Any]:
        return {
            "visited_maps": list(self.visited_maps),
            "visited_coords": list(self.visited_coords),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if "visited_maps" in state_dict:
            self.visited_maps = set(tuple(x) if isinstance(x, list) else x for x in state_dict["visited_maps"])
        if "visited_coords" in state_dict:
            self.visited_coords = set(tuple(x) if isinstance(x, list) else x for x in state_dict["visited_coords"])

    def reset(self):
        self.visited_maps.clear()
        self.visited_coords.clear()


class ItemReward(RewardFunction):
    """物品奖励 - 基于获得新物品"""

    def __init__(self, item_weight: float = 5.0):
        super().__init__("item")
        self.item_weight = item_weight

    def compute(self, prev_state: Dict, current_state: Dict, action: int) -> float:
        prev_items = self._count_items(prev_state)
        curr_items = self._count_items(current_state)

        diff = curr_items - prev_items
        return diff * self.item_weight

    def _count_items(self, state: Dict) -> int:
        bag = state.get("bag", {})
        total = 0
        for pocket in bag.values():
            total += len(pocket)
        return total


class StepPenalty(RewardFunction):
    """步数惩罚 - 鼓励快速完成任务"""

    def __init__(self, penalty: float = -0.1):
        super().__init__("step_penalty")
        self.penalty = penalty

    def compute(self, prev_state: Dict, current_state: Dict, action: int) -> float:
        return self.penalty


class ImageChangeReward(RewardFunction):
    """图像变化奖励 - 基于屏幕像素变化"""

    def __init__(self, change_weight: float = 0.1, threshold: int = 10000):
        super().__init__("image_change")
        self.change_weight = change_weight
        self.threshold = threshold
        self.prev_image = None

    def compute(self, prev_state: Dict, current_state: Dict, action: int) -> float:
        prev_img = prev_state.get("screenshot")
        curr_img = current_state.get("screenshot")

        if prev_img is None or curr_img is None:
            return 0.0

        # 降采样 (84x84 -> 42x42)
        from PIL import Image
        prev_small = prev_img.resize((42, 42), Image.BILINEAR)
        curr_small = curr_img.resize((42, 42), Image.BILINEAR)

        # 转换为 numpy 数组
        prev_arr = np.array(prev_small)
        curr_arr = np.array(curr_small)

        # 计算差异
        diff = np.abs(curr_arr.astype(int) - prev_arr.astype(int)).sum()

        if diff > self.threshold:
            return self.change_weight
        return 0.0


class SurvivalReward(RewardFunction):
    """生存奖励 - 每次存活给正奖励"""

    def __init__(self, reward: float = 0.1):
        super().__init__("survival")
        self.reward = reward

    def compute(self, prev_state: Dict, current_state: Dict, action: int) -> float:
        # 检查是否有存活的宝可梦
        party = current_state.get("party", [])
        alive = any(mon.get("hp", 0) > 0 for mon in party)
        return self.reward if alive else -100.0  # 团灭惩罚


class ScreenStaticPenalty(RewardFunction):
    """屏幕静止惩罚 - 如果屏幕没有变化则给予惩罚"""

    def __init__(self, penalty: float = -0.01, threshold: int = 10000):
        super().__init__("screen_static_penalty")
        self.penalty = penalty
        self.threshold = threshold

    def compute(self, prev_state: Dict, current_state: Dict, action: int) -> float:
        prev_img = prev_state.get("screenshot")
        curr_img = current_state.get("screenshot")

        if prev_img is None or curr_img is None:
            return 0.0

        # 降采样 (84x84 -> 42x42)
        from PIL import Image
        prev_small = prev_img.resize((42, 42), Image.BILINEAR)
        curr_small = curr_img.resize((42, 42), Image.BILINEAR)

        # 转换为 numpy 数组
        prev_arr = np.array(prev_small)
        curr_arr = np.array(curr_small)

        # 计算差异
        diff = np.abs(curr_arr.astype(int) - prev_arr.astype(int)).sum()

        # 如果差异小于阈值（屏幕基本静止），则给予惩罚
        if diff < self.threshold:
            return self.penalty
        return 0.0


class GameStartReward(RewardFunction):
    """
    开局引导奖励

    引导 Agent 完成：
    1. 开始游戏 (Press Start -> New Game)
    2. 完成对话和起名
    3. 进入游戏世界 (Littleroot Town)
    4. 获得初始宝可梦
    """

    def __init__(self, milestone_reward: float = 10.0):
        super().__init__("game_start")
        self.milestone_reward = milestone_reward
        self.milestones = {
            "in_game": False,      # 进入游戏世界（地图有效）
            "littleroot": False,   # 到达未白镇
            "route101": False,     # 到达 101 号道路
            "has_pokemon": False,  # 获得宝可梦
        }

        # 关键地图 ID (Group, Num)
        # 注意：这些 ID 需要根据实际 ROM 确认，这里使用常见值
        # Littleroot Town: (0, 9)
        # Route 101: (0, 16)
        # Inside Truck: (25, 40) ?
        # Player House 2F: (4, 1) ?
        self.target_maps = {
            (0, 9): "littleroot",
            (0, 16): "route101",
        }

    def compute(self, prev_state: Dict, current_state: Dict, action: int) -> float:
        reward = 0.0

        # 1. 检查是否进入游戏世界
        # 如果地图信息有效且不是 (0,0)，说明已经进入游戏
        loc = current_state.get("location", {})
        map_key = (loc.get("map_group", 0), loc.get("map_num", 0))

        if not self.milestones["in_game"]:
            if map_key != (0, 0):
                self.milestones["in_game"] = True
                reward += self.milestone_reward
                # print("Milestone reached: Entered Game World")

        # 2. 检查特定地图里程碑
        if self.milestones["in_game"]:
            milestone_name = self.target_maps.get(map_key)
            if milestone_name and not self.milestones.get(milestone_name, False):
                self.milestones[milestone_name] = True
                reward += self.milestone_reward
                # print(f"Milestone reached: Arrived at {milestone_name}")

        # 3. 检查是否获得宝可梦
        if not self.milestones["has_pokemon"]:
            party = current_state.get("party", [])
            if len(party) > 0:
                self.milestones["has_pokemon"] = True
                reward += self.milestone_reward * 5.0  # 大额奖励
                # print("Milestone reached: Got First Pokemon!")

        return reward

    def state_dict(self) -> Dict[str, Any]:
        return {"milestones": self.milestones}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        if "milestones" in state_dict:
            self.milestones = state_dict["milestones"]

    def reset(self):
        self.milestones = {
            "in_game": False,
            "littleroot": False,
            "route101": False,
            "has_pokemon": False,
        }


class CustomRewardFunction:
    """
    自定义奖励函数的工厂类

    使用示例:
        # 创建自定义奖励函数
        def my_reward_fn(prev_state, current_state, action):
            # 在这里编写你的奖励逻辑
            reward = 0.0

            # 示例: 奖励等级提升
            prev_levels = [p.get('level', 0) for p in prev_state.get('party', [])]
            curr_levels = [p.get('level', 0) for p in current_state.get('party', [])]
            level_up = sum(curr_levels) - sum(prev_levels)
            reward += level_up * 10.0

            # 示例: 惩罚死亡
            prev_hp = sum(p.get('hp', 0) for p in prev_state.get('party', []))
            curr_hp = sum(p.get('hp', 0) for p in current_state.get('party', []))
            if curr_hp == 0 and prev_hp > 0:
                reward -= 50.0

            return reward

        # 使用自定义奖励函数
        reward_fn = CustomRewardFunction.create(my_reward_fn)
    """

    @staticmethod
    def create(func: Callable[[Dict, Dict, int], float], name: str = "custom") -> RewardFunction:
        """将自定义函数包装为 RewardFunction"""

        class AnonymousReward(RewardFunction):
            def __init__(self):
                super().__init__(name)
                self.func = func

            def compute(self, prev_state: Dict, current_state: Dict, action: int) -> float:
                return self.func(prev_state, current_state, action)

        return AnonymousReward()

    @staticmethod
    def create_default() -> CombinedReward:
        """
        创建默认奖励函数

        组合了:
        - HP变化奖励
        - 升级奖励
        - 探索奖励 (地图 + 坐标)
        - 物品奖励
        - 步数惩罚 (轻微)
        - 屏幕静止惩罚
        - 屏幕变化奖励
        - 开局引导奖励
        """
        return CombinedReward({
            HPChangeReward(hp_weight=1.0): 0.0,
            LevelUpReward(level_weight=10.0): 0.0,
            ExplorationReward(location_weight=1.0, coord_weight=0.05): 1.0,
            ItemReward(item_weight=5.0): 0.0,
            StepPenalty(penalty=-0.01): 0.0,
            ScreenStaticPenalty(penalty=-0.1, threshold=10000): 1.0,
            ImageChangeReward(change_weight=0.1, threshold=10000): 1.0,
            GameStartReward(milestone_reward=10.0): 1.0,
        })


# 预定义的奖励函数配置
REWARD_CONFIGS = {
    "default": lambda: CustomRewardFunction.create_default(),
    "simple": lambda: CombinedReward({
        HPChangeReward(hp_weight=1.0): 1.0,
        StepPenalty(penalty=-0.01): 1.0,
    }),
    "battle": lambda: CombinedReward({
        HPChangeReward(hp_weight=2.0): 1.0,
        BattleWinReward(win_reward=100.0): 1.0,
        StepPenalty(penalty=-0.1): 1.0,
    }),
    "exploration": lambda: CombinedReward({
        ExplorationReward(location_weight=5.0): 1.0,
        StepPenalty(penalty=-0.01): 1.0,
    }),
    "start_game": lambda: CombinedReward({
        ScreenStaticPenalty(penalty=-0.1, threshold=10000): 1.0,
        ImageChangeReward(change_weight=0.1, threshold=10000): 1.0,
        GameStartReward(milestone_reward=20.0): 1.0,
        StepPenalty(penalty=-0.001): 1.0,
        ExplorationReward(location_weight=5.0, coord_weight=0.1): 1.0,
    }),
}

