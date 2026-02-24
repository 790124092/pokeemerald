# Pokemon Emerald RL Agent

This directory contains the code for training a Reinforcement Learning agent to play Pokemon Emerald.

## Directory Structure

- `environment/`: Contains the interface to the mGBA emulator.
  - `emulator.py`: The main class `Emulator` that wraps mGBA and provides state reading (Party, Bag, Location).
- `rl/`: Contains the RL model and training loop.
  - `model.py`: The PyTorch model (`PokemonAgent`) and state encoding logic.
  - `train.py`: The main training script.

## Setup

1. **Install Dependencies**:
   You need Python 3.8+ and the following packages:
   ```bash
   pip install torch numpy
   ```

2. **Install mGBA Python Bindings**:
   The default Homebrew installation of mGBA does NOT include Python bindings. You must build from source.

   **macOS (Build from Source):**
   ```bash
   # 1. Install dependencies
   brew install cmake libzip qt5 sdl2 libedit ffmpeg

   # 2. Clone mGBA repository
   git clone https://github.com/mgba-emu/mgba.git
   cd mgba

   # 3. Build with Python support
   mkdir build
   cd build
   cmake -DBUILD_PYTHON=ON ..
   make
   sudo make install
   ```

   After installation, verify it works:
   ```bash
   python3 -c "import mgba.core"
   ```

3. **ROM and Map File**:
   Ensure `pokeemerald.gba` and `pokeemerald.map` are in the root directory of the project.
   The `pokeemerald.map` file is generated during the build process (`make`).

## Usage

To run the training loop:

```bash
python3 agent/rl/train.py
```

To test the environment wrapper:

```bash
python3 agent/environment/emulator.py
```

## State Representation

The `Emulator.get_state()` method returns a dictionary with:
- `location`: Coordinates (x, y) and Map ID/Name.
- `party`: List of 6 Pokemon with stats, moves, IVs, EVs, etc.
- `bag`: Dictionary of items in each pocket.
- `screenshot`: The current game screen (if enabled).

## Notes

- The `Emulator` class uses memory addresses from `pokeemerald.map` to read game state directly from RAM.
- The `Pokemon` data structure is decrypted on-the-fly using the game's encryption algorithm.
- Text decoding (Nicknames, OT Names) uses a custom charmap loader (`charmap.txt`).

