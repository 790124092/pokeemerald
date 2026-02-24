"""
Test script to verify the RL environment works correctly (without gym)
"""

import os
import sys

# 设置库路径
os.environ['DYLD_LIBRARY_PATH'] = '/Users/konglingkai/codes/mgba/build/install/lib'

# 找到项目根目录
script_dir = os.path.dirname(os.path.abspath(__file__))
# agent/ 在 pokeemerald/ 下，所以项目根目录是 parent
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)
os.chdir(project_root)

# 直接导入 emulator 模块
from agent.environment.emulator import Emulator


def test_emulator():
    """测试 emulator 基本功能"""
    print("=" * 50)
    print("Testing Emulator")
    print("=" * 50)

    emu = Emulator("pokeemerald.gba")

    if not emu.core:
        print("Failed to create emulator!")
        return False

    print("Emulator created successfully!")

    # 运行几帧
    for i in range(10):
        emu.run_frame()

    # 获取状态
    state = emu.get_state(include_screenshot=True)

    print(f"\nLocation: {state.get('location', {})}")
    print(f"Party size: {len(state.get('party', []))}")
    print(f"Bag: {state.get('bag', {})}")

    # 打印队伍详情
    party = state.get('party', [])
    for i, mon in enumerate(party):
        print(f"  Pokemon {i+1}: {mon.get('species', 'Unknown')} Lv.{mon.get('level', 0)} HP:{mon.get('hp', 0)}/{mon.get('max_hp', 0)}")

    # 打印包裹详情
    bag = state.get('bag', {})
    for pocket_name, items in bag.items():
        if items:
            item_strs = [f"{item['name']} x{item['quantity']}" for item in items[:5]]
            print(f"  {pocket_name}: {', '.join(item_strs)}")

    if state.get('screenshot'):
        print(f"\nScreenshot size: {state['screenshot'].size}")

    return True


def test_reward_functions():
    """测试奖励函数"""
    print("\n" + "=" * 50)
    print("Testing Reward Functions")
    print("=" * 50)

    from agent.rl.reward_function import (
        CustomRewardFunction,
        REWARD_CONFIGS
    )

    # 测试默认奖励函数
    default_reward = REWARD_CONFIGS['default']()
    print(f"Default reward function: {default_reward.name}")

    # 测试自定义奖励函数
    def my_reward_fn(prev_state, current_state, action):
        # 简单的奖励：基于等级变化
        prev_levels = [p.get('level', 0) for p in prev_state.get('party', [])]
        curr_levels = [p.get('level', 0) for p in current_state.get('party', [])]
        return sum(curr_levels) - sum(prev_levels)

    custom_reward = CustomRewardFunction.create(my_reward_fn, "level_based")
    print(f"Custom reward function: {custom_reward.name}")

    # 测试奖励计算
    mock_prev_state = {'party': [{'level': 5}], 'location': {'map_name': 'test'}}
    mock_curr_state = {'party': [{'level': 6}], 'location': {'map_name': 'test2'}}

    reward = custom_reward.compute(mock_prev_state, mock_curr_state, 0)
    print(f"Test reward (level 5->6): {reward}")

    return True


def main():
    print("Pokemon RL Environment Test")
    print("=" * 50)

    # 测试 emulator
    if not test_emulator():
        print("Emulator test failed!")
        return

    # 测试奖励函数
    if not test_reward_functions():
        print("Reward function test failed!")
        return

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
    print("\nNote: To use the full Gym environment, install gym and stable-baselines3:")
    print("  pip install gym stable-baselines3")


if __name__ == '__main__':
    main()

