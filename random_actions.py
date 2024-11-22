# Perform random actions in the environment and create a GIF
from pathlib import Path

from pyboy.pyboy import PyBoy
from pyboy.utils import WindowEvent

from envs.mario_deluxe import MarioDeluxe

rom_path = Path("Super Mario Bros. Deluxe (U) (V1.1) [C][!].gbc")
pyboy = PyBoy(str(rom_path), sound=False)

env = MarioDeluxe(pyboy, n_frames=1, render=True, debug=False)
observation = env.reset()

env.pyboy.send_input(WindowEvent.SCREEN_RECORDING_TOGGLE)
for _ in range(1_000):
    action = env.action_space.sample()
    env.step(action)

env.pyboy.send_input(WindowEvent.SCREEN_RECORDING_TOGGLE)
env.step(0)
env.close()
