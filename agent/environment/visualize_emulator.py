import os
import sys

import pygame
from mgba._pylib import ffi, lib
from mgba.gba import GBA

from agent.environment.emulator import Emulator

# Save original stdout/stderr
original_stdout_fd = os.dup(sys.stdout.fileno())
original_stderr_fd = os.dup(sys.stderr.fileno())

# Redirect stdout to /dev/null to silence mgba logs
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, sys.stdout.fileno())
# os.dup2(devnull, sys.stderr.fileno())

# Create a new file object for original stdout
my_stdout = os.fdopen(original_stdout_fd, 'w')

def print_debug(*args, **kwargs):
    print(*args, file=my_stdout, **kwargs)
    my_stdout.flush()

# Disable mgba logging by setting default logger to NULL (just in case)
lib.mLogSetDefaultLogger(ffi.NULL)

# Key mapping
KEY_MAP = {
    pygame.K_x: GBA.KEY_A,
    pygame.K_z: GBA.KEY_B,
    pygame.K_a: GBA.KEY_L,
    pygame.K_s: GBA.KEY_R,
    pygame.K_RETURN: GBA.KEY_START,
    pygame.K_BACKSPACE: GBA.KEY_SELECT,
    pygame.K_UP: GBA.KEY_UP,
    pygame.K_DOWN: GBA.KEY_DOWN,
    pygame.K_LEFT: GBA.KEY_LEFT,
    pygame.K_RIGHT: GBA.KEY_RIGHT,
}

def main():
    pygame.init()

    # Initialize emulator
    emu = Emulator("../../pokeemerald.gba")
    if not emu.core:
        print_debug("Failed to initialize emulator")
        return

    # Set up display
    width, height = 240, 160
    scale = 3
    screen = pygame.display.set_mode((width * scale, height * scale))
    pygame.display.set_caption("PokeEmerald Emulator Visualization")

    clock = pygame.time.Clock()
    running = True

    print_debug("Controls:")
    print_debug("Arrow Keys: D-Pad")
    print_debug("X: A Button")
    print_debug("Z: B Button")
    print_debug("A: L Button")
    print_debug("S: R Button")
    print_debug("Enter: Start")
    print_debug("Backspace: Select")
    print_debug("F1: Save State (state.ss1)")
    print_debug("F2: Load State (state.ss1)")
    print_debug("ESC: Quit")

    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_F1:
                    if emu.save_state("state.ss1"):
                        print_debug("State saved to state.ss1")
                    else:
                        print_debug("Failed to save state")
                elif event.key == pygame.K_F2:
                    if emu.load_state("state.ss1"):
                        print_debug("State loaded from state.ss1")
                    else:
                        print_debug("Failed to load state")
                elif event.key in KEY_MAP:
                    print_debug(f"Key pressed: {pygame.key.name(event.key)} -> GBA Key: {KEY_MAP[event.key]}")
                    emu.press_button(KEY_MAP[event.key])
            elif event.type == pygame.KEYUP:
                if event.key in KEY_MAP:
                    print_debug(f"Key released: {pygame.key.name(event.key)}")
                    emu.release_button(KEY_MAP[event.key])

        # Run emulator frame
        # emu.core.set_video_buffer(emu.image)
        emu.run_frame()

        # Get video frame
        frame_image = emu.get_frame_buffer()

        # Test: Manually modify buffer to verify display logic
        if frame_image:
            # from mgba._pylib import ffi
            # Set first 100 pixels to white (assuming 32-bit color)
            # for i in range(100):
            #     frame_image.buffer[i] = 0xFFFFFFFF
            if not hasattr(main, "has_printed_dir"):
                print_debug(f"frame_image attributes: {dir(frame_image)}")
                main.has_printed_dir = True

        # Periodic debug print
        frame_count = getattr(main, "frame_count", 0)
        main.frame_count = frame_count + 1
        if main.frame_count % 60 == 0:
             if frame_image:
                 data = frame_image.tobytes()
                 non_zero = any(b != 0 for b in data)
                 print_debug(f"DEBUG: Frame {main.frame_count}, Image size: {frame_image.size}, Non-zero: {non_zero}")
             else:
                 print_debug(f"DEBUG: Frame {main.frame_count}, Image is None")

        if frame_image:
            # Convert PIL Image to Pygame Surface
            # PIL Image is RGBA, Pygame expects string/bytes
            mode = frame_image.mode
            size = frame_image.size
            data = frame_image.tobytes()

            # Debug: check if data is all zeros
            if not hasattr(main, "has_printed_debug"):
                print_debug("\n" + "="*50)
                print_debug(f"DEBUG: Image mode: {mode}, size: {size}, data len: {len(data)}")
                non_zero = any(b != 0 for b in data)
                print_debug(f"DEBUG: Image has non-zero data: {non_zero}")
                if non_zero:
                    print_debug(f"DEBUG: First 20 bytes: {list(data[:20])}")
                print_debug("="*50 + "\n")
                main.has_printed_debug = True

            py_image = pygame.image.frombytes(data, size, mode)

            # Scale and blit
            scaled_image = pygame.transform.scale(py_image, (width * scale, height * scale))
            screen.blit(scaled_image, (0, 0))
        else:
            if not hasattr(main, "has_printed_none"):
                print_debug("Frame image is None")
                main.has_printed_none = True

        pygame.display.flip()
        clock.tick(60) # Limit to 60 FPS

    pygame.quit()

    # Restore stdout/stderr (optional, as we are exiting)
    # os.dup2(original_stdout_fd, sys.stdout.fileno())
    # os.dup2(original_stderr_fd, sys.stderr.fileno())
    # os.close(original_stdout_fd)
    # os.close(original_stderr_fd)
    # os.close(devnull)

if __name__ == "__main__":
    main()

