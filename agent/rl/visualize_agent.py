"""
Visualize RL Agent

加载训练好的模型并展示其在游戏中的表现
"""

import argparse
import os
import sys

import pygame

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# os.chdir(project_root) # Don't change cwd, it might confuse relative paths if run from root

# Redirect C-level stdout/stderr to /dev/null to silence mgba logs
try:
    original_stdout_fd = os.dup(sys.stdout.fileno())
    original_stderr_fd = os.dup(sys.stderr.fileno())
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, sys.stdout.fileno())
    os.dup2(devnull, sys.stderr.fileno())
    sys.stdout = os.fdopen(original_stdout_fd, 'w')
    sys.stderr = os.fdopen(original_stderr_fd, 'w')
except Exception as e:
    print(f"Failed to silence logs: {e}", file=sys.stderr)

# Disable mgba logging
try:
    from mgba._pylib import ffi, lib
    lib.mLogSetDefaultLogger(ffi.NULL)
except ImportError:
    pass

from agent.rl.agent import PPOAgent, DQNAgent, RandomAgent
from agent.environment.pokemon_env import PokemonEnv


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize Pokemon RL Agent')

    parser.add_argument('--rom', type=str, default='pokeemerald.gba',
                        help='Path to Pokemon ROM')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--algorithm', type=str, default='ppo',
                        choices=['ppo', 'dqn', 'random'],
                        help='RL algorithm')
    parser.add_argument('--frame-skip', type=int, default=10,
                        help='Number of frames to skip per action')
    parser.add_argument('--image-size', type=int, nargs=2, default=[84, 84],
                        help='Image size for state representation')
    parser.add_argument('--fps', type=int, default=60,
                        help='Target FPS for visualization')
    parser.add_argument('--scale', type=int, default=3,
                        help='Display scale')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cuda/cpu)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize pygame
    pygame.init()
    width, height = 240, 160
    scale = args.scale
    screen = pygame.display.set_mode((width * scale, height * scale))
    pygame.display.set_caption(f"Pokemon Agent Visualization - {args.algorithm.upper()}")
    clock = pygame.time.Clock()

    # Create environment
    print(f"Creating environment with ROM: {args.rom}")
    env = PokemonEnv(
        rom_path=args.rom,
        frame_skip=args.frame_skip,
        image_size=tuple(args.image_size),
    )

    # Create agent
    n_actions = env.action_space.n
    print(f"Creating {args.algorithm.upper()} agent")

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

    # Load model
    if args.algorithm != 'random':
        print(f"Loading model from {args.model}")
        try:
            agent.load(args.model)
        except Exception as e:
            print(f"Failed to load model: {e}")
            return

    # Main loop
    running = True
    state = env.reset()
    total_reward = 0
    step = 0

    print("Starting visualization...")
    print("Press ESC to quit")

    while running:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # Select action
        action, _, _ = agent.select_action(state, training=False)
        action_name = env.ACTIONS[action]

        # Step environment
        state, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

        # Get full resolution screenshot for display
        # Note: env.current_state['screenshot'] contains the full res PIL image
        screenshot = env.current_state.get('screenshot')

        if screenshot:
            # Convert PIL Image to Pygame Surface
            mode = screenshot.mode
            size = screenshot.size
            data = screenshot.tobytes()

            py_image = pygame.image.frombytes(data, size, mode)
            scaled_image = pygame.transform.scale(py_image, (width * scale, height * scale))
            screen.blit(scaled_image, (0, 0))

            # Draw info
            font = pygame.font.SysFont("Arial", 16)

            # Action
            text = font.render(f"Action: {action_name}", True, (255, 255, 255))
            screen.blit(text, (10, 10))

            # Reward
            text = font.render(f"Reward: {reward:.2f} (Total: {total_reward:.2f})", True, (255, 255, 255))
            screen.blit(text, (10, 30))

            # Step
            text = font.render(f"Step: {step}", True, (255, 255, 255))
            screen.blit(text, (10, 50))

            # Location
            loc = info.get('location', {})
            map_name = loc.get('map_name', 'Unknown')
            text = font.render(f"Loc: {map_name}", True, (255, 255, 255))
            screen.blit(text, (10, 70))

        pygame.display.flip()
        clock.tick(args.fps)

        if done:
            print(f"Episode finished. Total reward: {total_reward:.2f}")
            state = env.reset()
            total_reward = 0
            step = 0

    pygame.quit()
    env.close()

if __name__ == "__main__":
    main()

