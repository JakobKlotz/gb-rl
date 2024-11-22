from gymnasium.wrappers import GrayScaleObservation
from pyboy import PyBoy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from envs.callback import TrainAndLoggingCallback
from envs.mario_deluxe import MarioDeluxe

TIMESTEPS, FRAMES = 1_000_000, 5
POLICY = "MlpPolicy"

pyboy = PyBoy(
    "Super Mario Bros. Deluxe (U) (V1.1) [C][!].gbc",
    sound=False,
    window="null",  # headless
)
env = MarioDeluxe(pyboy, policy=POLICY, render=False, n_frames=FRAMES)

# gray scale the RGB image
if POLICY == "CnnPolicy":
    env = GrayScaleObservation(env, keep_dim=True)

env = DummyVecEnv([lambda: env])
# add last 4 frames to the state (should improve the agent)
env = VecFrameStack(env, n_stack=4, channels_order="last")

agent = PPO(policy=POLICY, env=env, verbose=1, seed=42, n_steps=2048)
# add a callback to save the model every 100k steps
path = f"models/{POLICY}-frames-{FRAMES}"
agent.learn(
    progress_bar=True,
    total_timesteps=TIMESTEPS,
    callback=TrainAndLoggingCallback(
        check_freq=100_000,
        save_path=path,
        model_prefix=f"{POLICY}-ppo_mario_deluxe",
    ),
)
agent.save(f"{path}/ppo_mario_deluxe-final")
