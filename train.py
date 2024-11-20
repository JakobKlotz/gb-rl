from stable_baselines3 import PPO
from envs.mario_deluxe import MarioDeluxe
from pyboy import PyBoy

pyboy = PyBoy("Super Mario Bros. Deluxe (U) (V1.1) [C][!].gbc", sound=False)

env = MarioDeluxe(pyboy)
env.reset()

agent = PPO("MlpPolicy", env, verbose=1, seed=42, n_steps=1024)
agent.learn(progress_bar=True, total_timesteps=200_000)
agent.save("ppo_mario_deluxe")
