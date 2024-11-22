# Load actions from the best run (played by the agent) and create a GIF
import json
from pathlib import Path

from pyboy.pyboy import PyBoy

from envs.mario_deluxe import MarioDeluxe

# if the GIF should be in realtime, the environment must be rendered
# with 1 frame per step; instead of the 5 frames (which was used for training)
# each predicted action will therefore be performed 5 times
REALTIME: bool = True

rom_path = Path("game/Super Mario Bros. Deluxe (U) (V1.1) [C][!].gbc")
pyboy = PyBoy(str(rom_path), sound=False)

# load the best run
with Path("best_run.json").open("r") as f:
    best_run = json.load(f)

env = MarioDeluxe(
    pyboy, render=True, n_frames=1 if REALTIME else 5, debug=True
)
env.reset()

env.toggle_record()
for action in best_run["actions"]:
    if REALTIME:
        # perform each predicted action 5 times
        for _ in range(5):
            env.step(action)

    else:
        env.step(action)

env.toggle_record()
env.pyboy.tick()

env.close()
