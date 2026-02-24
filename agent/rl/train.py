"""
Pokemon RL Training Script

使用 PPO 算法训练 Pokemon 游戏 agent
"""

import argparse
import json
import os
import sys
from collections import deque
from datetime import datetime

import numpy as np
import torch
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn

# Redirect C-level stdout/stderr to /dev/null to silence mgba logs
# while keeping Python's sys.stdout/sys.stderr pointing to the original streams
try:
    # Save original file descriptors
    original_stdout_fd = os.dup(sys.stdout.fileno())
    original_stderr_fd = os.dup(sys.stderr.fileno())

    # Open /dev/null
    devnull = os.open(os.devnull, os.O_WRONLY)

    # Redirect stdout/stderr (fd 1 and 2) to /dev/null
    os.dup2(devnull, sys.stdout.fileno())
    os.dup2(devnull, sys.stderr.fileno())

    # Restore sys.stdout/sys.stderr to point to the original file descriptors
    sys.stdout = os.fdopen(original_stdout_fd, 'w')
    sys.stderr = os.fdopen(original_stderr_fd, 'w')
except Exception as e:
    print(f"Failed to silence logs: {e}", file=sys.stderr)

# Disable mgba logging (extra measure)
try:
    from mgba._pylib import ffi, lib
    lib.mLogSetDefaultLogger(ffi.NULL)
except ImportError:
    pass

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.chdir(project_root)

from agent.rl.agent import PPOAgent, DQNAgent, RandomAgent
from agent.rl.reward_function import (
    REWARD_CONFIGS,
    CustomRewardFunction,
    ScreenStaticPenalty
)
from agent.environment.pokemon_env import PokemonEnv


def parse_args():
    parser = argparse.ArgumentParser(description='Train Pokemon RL Agent')

    # 环境参数
    parser.add_argument('--rom', type=str, default='pokeemerald.gba',
                        help='Path to Pokemon ROM')
    parser.add_argument('--frame-skip', type=int, default=10,
                        help='Number of frames to skip per action')
    parser.add_argument('--image-size', type=int, nargs=2, default=[84, 84],
                        help='Image size for state representation')

    # 训练参数
    parser.add_argument('--algorithm', type=str, default='ppo',
                        choices=['ppo', 'dqn', 'random'],
                        help='RL algorithm')
    parser.add_argument('--total-steps', type=int, default=1000000,
                        help='Total training steps')
    parser.add_argument('--update-freq', type=int, default=1000,
                        help='Update policy every N steps')
    parser.add_argument('--save-freq', type=int, default=10000,
                        help='Save model every N steps')
    parser.add_argument('--log-freq', type=int, default=100,
                        help='Log every N steps')
    parser.add_argument('--eval-freq', type=int, default=5000,
                        help='Evaluate every N steps')

    # 奖励函数
    parser.add_argument('--reward-config', type=str, default='default',
                        choices=list(REWARD_CONFIGS.keys()),
                        help='Reward function configuration')
    parser.add_argument('--custom-reward', type=str, default=None,
                        help='Path to custom reward function file')

    # 模型参数
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--eps-clip', type=float, default=0.2,
                        help='PPO clip range')

    # 其他
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Directory to save models')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    # 自动检测设备
    default_device = "cpu"
    if torch.cuda.is_available():
        default_device = "cuda"
    elif torch.backends.mps.is_available():
        default_device = "mps"

    parser.add_argument('--device', type=str, default=default_device,
                        help='Device (cuda/mps/cpu)')

    return parser.parse_args()


def load_custom_reward_function(path: str):
    """加载自定义奖励函数"""
    # 这是一个示例，你需要根据实际情况实现
    exec_globals = {}
    exec(open(path).read(), exec_globals)
    reward_fn = exec_globals.get('reward_function')
    if reward_fn:
        return CustomRewardFunction.create(reward_fn)
    return None


def make_layout() -> Layout:
    """Define the layout."""
    layout = Layout(name="root")

    layout.split(
        Layout(name="header", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=10),
    )
    layout["main"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="center", ratio=1),
        Layout(name="right", ratio=1),
    )
    return layout


def train():
    args = parse_args()
    console = Console()

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 创建环境
    console.print(f"[bold green]Creating environment with ROM: {args.rom}[/bold green]")
    env = PokemonEnv(
        rom_path=args.rom,
        frame_skip=args.frame_skip,
        image_size=tuple(args.image_size),
    )

    # 设置奖励函数
    console.print(f"[bold blue]Using reward config: {args.reward_config}[/bold blue]")
    if args.custom_reward:
        reward_fn = load_custom_reward_function(args.custom_reward)
    else:
        reward_fn = REWARD_CONFIGS[args.reward_config]()
    env.set_reward_function(reward_fn)

    # 创建 agent
    console.print(f"[bold magenta]Creating {args.algorithm.upper()} agent[/bold magenta]")
    n_actions = env.action_space.n

    if args.algorithm == 'ppo':
        agent = PPOAgent(
            image_shape=(args.image_size[0], args.image_size[1], 3),
            state_dim=50,
            n_actions=n_actions,
            lr=args.lr,
            gamma=args.gamma,
            eps_clip=args.eps_clip,
            device=args.device,
        )
    elif args.algorithm == 'dqn':
        agent = DQNAgent(
            image_shape=(args.image_size[0], args.image_size[1], 3),
            state_dim=50,
            n_actions=n_actions,
            lr=args.lr,
            gamma=args.gamma,
            device=args.device,
        )
    elif args.algorithm == 'random':
        agent = RandomAgent(n_actions)
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    # 加载检查点（如果指定）
    if args.resume:
        console.print(f"[yellow]Resuming from {args.resume}[/yellow]")
        agent.load(args.resume)

    # 训练循环
    state = env.reset()
    episode_reward = 0
    episode_steps = 0
    episode_count = 0
    recent_rewards = deque(maxlen=100)
    log_messages = deque(maxlen=10)

    total_steps = 0
    log_episode_rewards = []

    layout = make_layout()

    # Header
    header_table = Table.grid(expand=True)
    header_table.add_column(justify="center", ratio=1)
    header_table.add_column(justify="right")
    header_table.add_row(
        f"[b]Pokemon RL Training[/b] - {args.algorithm.upper()} - {args.reward_config}",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    layout["header"].update(Panel(header_table, style="white on blue"))

    # Progress Bar
    job_progress = Progress(
        "{task.description}",
        SpinnerColumn(),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("{task.completed}/{task.total}"),
    )
    task_id = job_progress.add_task("[green]Training...", total=args.total_steps)

    with Live(layout, refresh_per_second=4, screen=True):
        while total_steps < args.total_steps:
            # 选择动作
            action, log_prob, value = agent.select_action(state, training=True)

            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 存储 transition
            if args.algorithm == 'ppo':
                agent.store_transition(state, action, reward, done, log_prob, value)
            elif args.algorithm == 'dqn':
                agent.store_transition(state, action, reward, next_state, done)

            # 更新统计
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            job_progress.update(task_id, advance=1)

            # 更新策略
            if total_steps % args.update_freq == 0:
                if args.algorithm == 'ppo':
                    agent.update()

            # 更新界面
            if total_steps % args.log_freq == 0:
                stats = agent.get_stats() if hasattr(agent, 'get_stats') else {}
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0

                # Left Panel: Current Episode
                left_table = Table(expand=True)
                left_table.add_column("Metric", style="cyan")
                left_table.add_column("Value", style="magenta")
                left_table.add_row("Episode", str(episode_count))
                left_table.add_row("Steps", str(episode_steps))
                left_table.add_row("Current Reward", f"{episode_reward:.2f}")

                if args.algorithm == 'dqn':
                    left_table.add_row("Epsilon", f"{getattr(agent, 'epsilon', 'N/A'):.4f}")
                elif args.algorithm == 'ppo':
                    entropy = stats.get('entropy', 0.0)
                    left_table.add_row("Entropy", f"{entropy:.4f}")
                    left_table.add_row("Policy Loss", f"{stats.get('policy_loss', 0.0):.4f}")
                    left_table.add_row("Value Loss", f"{stats.get('value_loss', 0.0):.4f}")

                layout["left"].update(Panel(left_table, title="Training Status", border_style="green"))

                # Center Panel: Game State
                center_table = Table(expand=True)
                center_table.add_column("State", style="cyan")
                center_table.add_column("Info", style="white")

                # Location
                loc = info.get('location', {})
                map_name = loc.get('map_name', 'Unknown')
                map_id = f"({loc.get('map_group', 0)}, {loc.get('map_num', 0)})"
                coords = f"({loc.get('x', 0)}, {loc.get('y', 0)})"
                center_table.add_row("Location", f"{map_name}\n{map_id} {coords}")

                # Party
                party = info.get('party', [])
                party_info = []
                for mon in party:
                    name = mon.get('species', 'Unknown')
                    lvl = mon.get('level', 0)
                    hp = f"{mon.get('hp', 0)}/{mon.get('max_hp', 0)}"
                    party_info.append(f"{name} Lv.{lvl} ({hp})")

                if not party_info:
                    center_table.add_row("Party", "Empty")
                else:
                    center_table.add_row("Party", "\n".join(party_info))

                # Bag (Summary)
                bag = info.get('bag', {})
                bag_summary = []
                for pocket, items in bag.items():
                    count = len(items)
                    if count > 0:
                        bag_summary.append(f"{pocket}: {count}")

                if bag_summary:
                    center_table.add_row("Bag", ", ".join(bag_summary))

                layout["center"].update(Panel(center_table, title="Game State", border_style="yellow"))

                # Right Panel: Statistics
                right_table = Table(expand=True)
                right_table.add_column("Metric", style="cyan")
                right_table.add_column("Value", style="yellow")
                right_table.add_row("Total Steps", f"{total_steps}/{args.total_steps}")
                right_table.add_row("Avg Reward (100)", f"{avg_reward:.2f}")
                if recent_rewards:
                    right_table.add_row("Max Reward (100)", f"{max(recent_rewards):.2f}")
                    right_table.add_row("Min Reward (100)", f"{min(recent_rewards):.2f}")

                layout["right"].update(Panel(right_table, title="Statistics", border_style="blue"))

                # Footer: Logs
                log_text = "\n".join(log_messages)
                layout["footer"].update(Panel(log_text, title="Logs", border_style="white"))

            # 保存模型
            if total_steps % args.save_freq == 0:
                save_path = os.path.join(args.save_dir, f'checkpoint_{total_steps}.pt')
                agent.save(save_path)
                log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Saved checkpoint to {save_path}")

            # 重置 episode
            if done:
                recent_rewards.append(episode_reward)
                log_episode_rewards.append(episode_reward)
                log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Episode {episode_count} finished. Reward: {episode_reward:.2f}")

                state = env.reset()
                episode_reward = 0
                episode_steps = 0
                episode_count += 1

    # 保存最终模型
    final_path = os.path.join(args.save_dir, 'final_model.pt')
    agent.save(final_path)
    print(f"Training complete! Final model saved to {final_path}")

    # 保存训练日志
    log_path = os.path.join(args.save_dir, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump({
            'episode_rewards': log_episode_rewards,
            'total_steps': total_steps,
            'args': vars(args),
        }, f, indent=2)
    print(f"Training log saved to {log_path}")

    env.close()


def evaluate():
    """评估模式"""
    args = parse_args()

    # 创建环境
    env = PokemonEnv(
        rom_path=args.rom,
        frame_skip=args.frame_skip,
        image_size=tuple(args.image_size),
    )

    # 加载奖励函数
    if args.custom_reward:
        reward_fn = load_custom_reward_function(args.custom_reward)
    else:
        reward_fn = REWARD_CONFIGS[args.reward_config]()
    env.set_reward_function(reward_fn)

    # 创建 agent
    n_actions = env.action_space.n

    if args.algorithm == 'ppo':
        agent = PPOAgent(
            image_shape=(args.image_size[0], args.image_size[1], 3),
            state_dim=50,
            n_actions=n_actions,
            device=args.device,
        )
    elif args.algorithm == 'dqn':
        agent = DQNAgent(
            image_shape=(args.image_size[0], args.image_size[1], 3),
            state_dim=50,
            n_actions=n_actions,
            device=args.device,
        )
    elif args.algorithm == 'random':
        agent = RandomAgent(n_actions)

    # 加载模型
    if args.resume:
        print(f"Loading model from {args.resume}")
        agent.load(args.resume)

    # 评估
    n_episodes = 10
    total_rewards = []

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        max_steps = 10000

        while steps < max_steps:
            action, _, _ = agent.select_action(state, training=False)
            state, reward, done, info = env.step(action)
            episode_reward += reward
            steps += 1

            if done:
                break

        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")

    print(f"\nAverage reward: {np.mean(total_rewards):.2f} +/- {np.std(total_rewards):.2f}")

    env.close()


def create_custom_reward_example():
    """创建自定义奖励函数示例"""
    example_code = '''
def custom_reward_function(prev_state, current_state, action):
    """
    自定义奖励函数

    Args:
        prev_state: 执行动作前的状态
        current_state: 执行动作后的状态
        action: 执行的动作

    Returns:
        reward: 奖励值
    """
    reward = 0.0

    # 示例 1: 奖励等级提升
    prev_levels = [p.get("level", 0) for p in prev_state.get("party", [])]
    curr_levels = [p.get("level", 0) for p in current_state.get("party", [])]
    level_up = sum(curr_levels) - sum(prev_levels)
    reward += level_up * 10.0

    # 示例 2: 奖励获得新物品
    prev_items = sum(len(p) for p in prev_state.get("bag", {}).values())
    curr_items = sum(len(p) for p in current_state.get("bag", {}).values())
    new_items = curr_items - prev_items
    reward += new_items * 5.0

    # 示例 3: 奖励探索新地图
    prev_loc = prev_state.get("location", {}).get("map_name", "")
    curr_loc = current_state.get("location", {}).get("map_name", "")
    if curr_loc != prev_loc:
        reward += 1.0

    # 示例 4: 惩罚 HP 损失
    prev_hp = sum(p.get("hp", 0) for p in prev_state.get("party", []))
    curr_hp = sum(p.get("hp", 0) for p in current_state.get("party", []))
    hp_diff = curr_hp - prev_hp
    if hp_diff < 0:
        reward += hp_diff * 2.0  # 惩罚损失

    # 示例 5: 奖励战斗胜利（需要根据游戏状态判断）
    # if current_state.get("battle_won", False):
    #     reward += 100.0

    return reward

# 然后使用 CustomRewardFunction 创建
from agent.reward_function import CustomRewardFunction
reward_fn = CustomRewardFunction.create(custom_reward_function, "my_reward")
'''

    example_path = os.path.join(os.path.dirname(__file__), 'custom_reward_example.py')
    with open(example_path, 'w') as f:
        f.write(example_code)
    print(f"Custom reward example saved to {example_path}")


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--create-example':
        create_custom_reward_example()
    elif len(sys.argv) > 1 and sys.argv[1] == '--evaluate':
        # 评估模式
        args = sys.argv[2:]
        sys.argv = ['train.py', '--evaluate'] + args
        evaluate()
    else:
        train()

