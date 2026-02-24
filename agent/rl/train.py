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
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.table import Table

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
    CustomRewardFunction
)
from agent.environment.pokemon_env import PokemonEnv
from agent.rl.vec_env import SubprocVecEnv


def parse_args():
    parser = argparse.ArgumentParser(description='Train Pokemon RL Agent')

    # 环境参数
    parser.add_argument('--rom', type=str, default='pokeemerald.gba',
                        help='Path to Pokemon ROM')
    parser.add_argument('--frame-skip', type=int, default=10,
                        help='Number of frames to skip per action')
    parser.add_argument('--image-size', type=int, nargs=2, default=[84, 84],
                        help='Image size for state representation')
    parser.add_argument('--num-envs', type=int, default=1,
                        help='Number of parallel environments')

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
    parser.add_argument('--ent-coef', type=float, default=0.05,
                        help='Entropy coefficient for PPO')

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

    parser.add_argument('--render', action='store_true',
                        help='Render game screen during training')

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


def make_env(rom_path, frame_skip, image_size, reward_fn=None):
    """创建环境的辅助函数"""
    def _init():
        env = PokemonEnv(
            rom_path=rom_path,
            frame_skip=frame_skip,
            image_size=image_size,
        )
        if reward_fn:
            env.set_reward_function(reward_fn)
        return env
    return _init


def train():
    args = parse_args()
    console = Console()

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 加载奖励函数
    console.print(f"[bold blue]Using reward config: {args.reward_config}[/bold blue]")
    if args.custom_reward:
        reward_fn = load_custom_reward_function(args.custom_reward)
    else:
        reward_fn = REWARD_CONFIGS[args.reward_config]()

    # 创建环境
    console.print(f"[bold green]Creating {args.num_envs} environment(s) with ROM: {args.rom}[/bold green]")

    if args.num_envs > 1:
        env_fns = [
            make_env(args.rom, args.frame_skip, tuple(args.image_size), reward_fn)
            for _ in range(args.num_envs)
        ]
        env = SubprocVecEnv(env_fns)
        # 获取 action space (假设所有环境一致)
        # 我们需要临时创建一个环境来获取 action space
        temp_env = PokemonEnv(rom_path=args.rom)
        n_actions = temp_env.action_space.n
        temp_env.close()
    else:
        env = PokemonEnv(
            rom_path=args.rom,
            frame_skip=args.frame_skip,
            image_size=tuple(args.image_size),
        )
        env.set_reward_function(reward_fn)
        n_actions = env.action_space.n

    # 创建 agent
    console.print(f"[bold magenta]Creating {args.algorithm.upper()} agent[/bold magenta]")

    if args.algorithm == 'ppo':
        agent = PPOAgent(
            image_shape=(args.image_size[0], args.image_size[1], 3),
            state_dim=50,
            n_actions=n_actions,
            lr=args.lr,
            gamma=args.gamma,
            eps_clip=args.eps_clip,
            ent_coef=args.ent_coef,
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

        # 尝试加载奖励函数状态
        # 假设 resume 路径是 checkpoints/checkpoint_X.pt
        # 奖励状态路径应该是 checkpoints/checkpoint_X_reward.json
        reward_state_path = args.resume.replace('.pt', '_reward.json')
        if os.path.exists(reward_state_path):
            console.print(f"[yellow]Loading reward state from {reward_state_path}[/yellow]")
            with open(reward_state_path, 'r') as f:
                reward_states = json.load(f)

            if args.num_envs > 1:
                if isinstance(reward_states, list) and len(reward_states) == args.num_envs:
                    env.set_reward_state(reward_states)
                else:
                    console.print("[red]Warning: Reward state mismatch with num_envs, skipping load[/red]")
            else:
                env.set_reward_state(reward_states)
        else:
            console.print("[yellow]No reward state file found, starting with fresh reward state[/yellow]")

    # 初始化 Pygame (如果需要渲染)
    screen = None
    if args.render:
        try:
            # Force driver for macOS
            if sys.platform == 'darwin':
                os.environ['SDL_VIDEODRIVER'] = 'cocoa'

            import pygame
            pygame.init()

            # Create window
            screen = pygame.display.set_mode((240 * 2, 160 * 2))  # 2x scale
            pygame.display.set_caption("Pokemon RL Training")

        except Exception as e:
            console.print(f"[red]Pygame init failed: {e}[/red]")

    # 训练循环
    state = env.reset()

    # 初始渲染
    if args.render and screen:
        try:
            import pygame
            # 如果是 VecEnv，state['image'] 是 (N, H, W, C)，取第一个
            if args.num_envs > 1:
                # VecEnv 的 state 已经是 dict of arrays
                # 我们需要从 info 中获取 screenshot，但 reset 只返回 state
                # 所以这里暂时用 state['image'] 或者跳过
                pass
            else:
                img = env.current_state.get('screenshot')
                if img:
                    mode = img.mode
                    size = img.size
                    data = img.tobytes()
                    py_image = pygame.image.fromstring(data, size, mode)
                    py_image = pygame.transform.scale(py_image, (240 * 2, 160 * 2))
                    screen.blit(py_image, (0, 0))
                    pygame.display.flip()
        except Exception as e:
            pass

    # 统计变量
    current_episode_rewards = np.zeros(args.num_envs)
    current_episode_steps = np.zeros(args.num_envs)

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
        f"[b]Pokemon RL Training[/b] - {args.algorithm.upper()} - {args.reward_config} - {args.num_envs} Envs",
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
            # 处理 Pygame 事件
            if args.render:
                import pygame
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return

            # 选择动作
            action, log_prob, value = agent.select_action(state, training=True)

            # 执行动作
            next_state, reward, done, info = env.step(action)

            # 更新统计
            # reward 和 done 都是数组 (N,)
            if args.num_envs == 1:
                # 单环境兼容
                reward = np.array([reward])
                done = np.array([done])
                info = [info] # Make it a list
                # action, log_prob, value 已经是 scalar 或 array，取决于 select_action
                # 如果是单环境，select_action 返回的是 scalar，需要转为 array 以便统一处理
                if not isinstance(action, np.ndarray):
                    action = np.array([action])
                    log_prob = np.array([log_prob])
                    value = np.array([value])

            current_episode_rewards += reward
            current_episode_steps += 1

            # 检查完成的 episode
            for i in range(args.num_envs):
                if done[i]:
                    recent_rewards.append(current_episode_rewards[i])
                    log_episode_rewards.append(current_episode_rewards[i])
                    log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Env {i} finished. Reward: {current_episode_rewards[i]:.2f}")

                    current_episode_rewards[i] = 0
                    current_episode_steps[i] = 0
                    episode_count += 1

            # 渲染画面 (显示所有环境的网格)
            if args.render and screen:
                try:
                    import pygame

                    # 获取所有环境的图像
                    images = []
                    if args.num_envs > 1:
                        for i in range(args.num_envs):
                            img = info[i].get('screenshot')
                            if img:
                                images.append(img)
                    else:
                        img = info[0].get('screenshot')
                        if img:
                            images.append(img)

                    if images:
                        # 计算网格布局
                        n_envs = len(images)
                        cols = int(np.ceil(np.sqrt(n_envs)))
                        rows = int(np.ceil(n_envs / cols))

                        # 单个图像原始大小 (2x scale)
                        single_w, single_h = 240 * 2, 160 * 2

                        # 完整拼接图的大小
                        full_w = single_w * cols
                        full_h = single_h * rows

                        # 屏幕最大尺寸限制 (适应 Mac 屏幕)
                        MAX_WIDTH = 1400
                        MAX_HEIGHT = 900

                        # 计算缩放比例
                        scale = 1.0
                        if full_w > MAX_WIDTH or full_h > MAX_HEIGHT:
                            scale = min(MAX_WIDTH / full_w, MAX_HEIGHT / full_h)

                        final_w = int(full_w * scale)
                        final_h = int(full_h * scale)

                        # 调整屏幕大小 (如果需要)
                        if screen.get_width() != final_w or screen.get_height() != final_h:
                            screen = pygame.display.set_mode((final_w, final_h))

                        # 创建一个大画布用于拼接
                        full_surface = pygame.Surface((full_w, full_h))

                        # 绘制每个环境到大画布
                        for i, img in enumerate(images):
                            # Convert PIL image to surface
                            mode = img.mode
                            size = img.size
                            data = img.tobytes()
                            py_image = pygame.image.fromstring(data, size, mode)
                            # Scale up to single cell size
                            py_image = pygame.transform.scale(py_image, (single_w, single_h))

                            # 计算位置
                            r = i // cols
                            c = i % cols
                            full_surface.blit(py_image, (c * single_w, r * single_h))

                        # 缩放并显示到屏幕
                        if scale != 1.0:
                            # 使用 smoothscale 获得更好的缩放质量
                            final_surface = pygame.transform.smoothscale(full_surface, (final_w, final_h))
                            screen.blit(final_surface, (0, 0))
                        else:
                            screen.blit(full_surface, (0, 0))

                        pygame.display.flip()
                    else:
                        if total_steps % 100 == 0:
                            log_messages.append(f"Warning: Screenshot is None at step {total_steps}")

                    # 保持窗口响应
                    pygame.event.pump()

                except Exception as e:
                    log_messages.append(f"Render Error: {str(e)}")

            # 存储 transition
            if args.algorithm == 'ppo':
                agent.store_transition(state, action, reward, done, log_prob, value)
            elif args.algorithm == 'dqn':
                # DQN 需要修改以支持 batch，暂时只支持单环境或需要修改 store_transition
                if args.num_envs == 1:
                    agent.store_transition(state, action[0], reward[0], next_state, done[0])
                else:
                    # TODO: Support batch DQN
                    pass

            state = next_state
            total_steps += 1
            job_progress.update(task_id, advance=1)

            # 更新策略
            if total_steps % args.update_freq == 0:
                if args.algorithm == 'ppo':
                    log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Updating policy...")
                    # Force a UI refresh before the blocking update
                    layout["footer"].update(Panel("\n".join(log_messages), title="Logs", border_style="white"))
                    # We can't easily force a Rich Live refresh here without access to the Live object directly
                    # But updating the layout object should be reflected on next refresh

                    agent.update()
                    log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Policy updated.")

            # 更新界面
            if total_steps % args.log_freq == 0:
                stats = agent.get_stats() if hasattr(agent, 'get_stats') else {}
                avg_reward = np.mean(recent_rewards) if recent_rewards else 0

                # Left Panel: Current Episode
                left_table = Table(expand=True)
                left_table.add_column("Metric", style="cyan")
                left_table.add_column("Value", style="magenta")
                left_table.add_row("Episodes", str(episode_count))
                left_table.add_row("Avg Reward (100)", f"{avg_reward:.2f}")

                if args.algorithm == 'dqn':
                    left_table.add_row("Epsilon", f"{getattr(agent, 'epsilon', 'N/A'):.4f}")
                elif args.algorithm == 'ppo':
                    entropy = stats.get('entropy', 0.0)
                    left_table.add_row("Entropy", f"{entropy:.4f}")
                    left_table.add_row("Policy Loss", f"{stats.get('policy_loss', 0.0):.4f}")
                    left_table.add_row("Value Loss", f"{stats.get('value_loss', 0.0):.4f}")

                layout["left"].update(Panel(left_table, title="Training Status", border_style="green"))

                # Center Panel: Game State (Best Env)
                center_table = Table(expand=True)
                center_table.add_column("State", style="cyan")
                center_table.add_column("Info", style="white")

                # Find best env (highest total party level)
                best_env_idx = 0
                max_total_level = -1

                # Ensure info is a list
                info_list = info if isinstance(info, (list, tuple)) else [info]

                for i, env_info in enumerate(info_list):
                    party = env_info.get('party', [])
                    total_level = sum(mon.get('level', 0) for mon in party)
                    if total_level > max_total_level:
                        max_total_level = total_level
                        best_env_idx = i

                best_info = info_list[best_env_idx]

                # Location
                loc = best_info.get('location', {})
                map_name = loc.get('map_name', 'Unknown')
                map_id = f"({loc.get('map_group', 0)}, {loc.get('map_num', 0)})"
                coords = f"({loc.get('x', 0)}, {loc.get('y', 0)})"
                center_table.add_row("Location", f"{map_name}\n{map_id} {coords}")

                # Party
                party = best_info.get('party', [])
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
                bag = best_info.get('bag', {})
                bag_summary = []
                for pocket, items in bag.items():
                    count = len(items)
                    if count > 0:
                        bag_summary.append(f"{pocket}: {count}")

                if bag_summary:
                    center_table.add_row("Bag", ", ".join(bag_summary))

                layout["center"].update(Panel(center_table, title=f"Game State (Env {best_env_idx} - Best)", border_style="yellow"))

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

                # 保存奖励函数状态
                reward_state_path = os.path.join(args.save_dir, f'checkpoint_{total_steps}_reward.json')
                if args.num_envs > 1:
                    # 对于多环境，我们保存所有环境的奖励状态列表
                    reward_states = env.get_reward_state()
                    # Convert sets to lists for JSON serialization
                    # Note: This assumes reward state is JSON serializable or we need to handle it
                    # Our current implementation returns dicts with lists, so it should be fine
                    with open(reward_state_path, 'w') as f:
                        json.dump(reward_states, f)
                else:
                    reward_state = env.get_reward_state()
                    with open(reward_state_path, 'w') as f:
                        json.dump(reward_state, f)

                log_messages.append(f"[{datetime.now().strftime('%H:%M:%S')}] Saved checkpoint to {save_path}")

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
            # 评估模式下，select_action 返回 scalar
            action, _, _ = agent.select_action(state, training=False)
            # 如果 select_action 返回的是 array (因为我们修改了它)，取第一个
            if isinstance(action, np.ndarray):
                action = action[0]

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

