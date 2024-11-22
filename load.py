# Load a model and play a couple of games ->
# save a run if the agent reached the end of the level
import json
from pathlib import Path

from gymnasium.wrappers import GrayScaleObservation
from pyboy import PyBoy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from tqdm import tqdm

from envs.mario_deluxe import MarioDeluxe

POLICY = "MlpPolicy"

model = PPO.load("models/Mlp-frames-5/MlpPolicy-ppo_mario_deluxe-final.zip")

pyboy = PyBoy(
    "game/Super Mario Bros. Deluxe (U) (V1.1) [C][!].gbc",
    sound=False,
)
env = MarioDeluxe(pyboy, policy=POLICY, render=True)
if POLICY == "CnnPolicy":
    env = GrayScaleObservation(env, keep_dim=True)
env = DummyVecEnv([lambda: env])
# add last 4 frames to the state (should improve the agent)
env = VecFrameStack(env, n_stack=4, channels_order="last")

best_run = {}
# couple of tries to reach the end of the level
for run in tqdm(range(100), desc="Games"):
    state = env.reset()
    done, info, level, player_x, actions = False, [], 0, 0, []

    while not done:
        action, _ = model.predict(state)
        actions.append(int(action[0]))

        state, reward, done, info = env.step(action)
        flag_reached, player_x, level = (
            info[0]["flag_reached"],
            info[0]["x_position"],
            info[0]["level_reached"],
        )

    if done and level > 0:
        # if the agent passed the first level, save the run
        best_run = {
            "run": run,
            "actions": actions,
            "x": player_x,
        }

# dump best run to JSON
with Path("best_run.json").open("w") as f:
    json.dump(best_run, f, indent=4)
