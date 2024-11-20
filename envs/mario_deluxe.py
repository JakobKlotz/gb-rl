# Adopted from https://github.com/Baekalfen/PyBoy/wiki/Using-PyBoy-with-Gym
import gymnasium as gym
from gymnasium import spaces
import numpy as np



class MarioDeluxe(gym.Env):
    def __init__(self, pyboy, policy: str = "MlpPolicy", debug=False):
        super().__init__()

        if policy not in ("MlpPolicy", "CnnPolicy"):
            raise ValueError("Policy must be 'MlpPolicy' or 'CnnPolicy'")

        self.pyboy = pyboy
        self.policy = policy
        self.matrix_shape = (
            (32, 32) if policy == "MlpPolicy" else (144, 160, 4)
        )
        self.actions = [
            "",
            "a",
            "b",
            "left",
            "right",]
        self._fitness = 0
        self._previous_fitness = 0
        self.debug = debug

        # to 'support' continuous button presses and
        # simultaneous button presses
        self.current_action = 0 # no action ""
        self.action_duration = 0
        self.max_action_duration = 5  # number of frames to hold a button

        if not self.debug:
            self.pyboy.set_emulation_speed(0)

        self.action_space = spaces.Discrete(len(self.actions))

        self.observation_space = spaces.Box(
            low=0, high=255, shape=self.matrix_shape, dtype=np.uint8
        )

        self.pyboy.game_wrapper.start_game()

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        # Move the agent
        # Handle multi-frame button press
        if action == 0:  # No action
            # pass
            self.current_action = 0# NOne
            self.action_duration = 0
        elif action != self.current_action:
            print(self.current_action, action)
            if self.current_action != 0:
            # New action, reset duration and press button
                self.pyboy.button_release(self.actions[self.current_action])

            self.current_action = action
            self.action_duration = 1
            self.pyboy.button_press(self.actions[action])
        else:
            # Continue existing action
            self.action_duration += 1

            # Release button if max duration reached
            if self.action_duration >= self.max_action_duration:
                self.pyboy.button_release(self.actions[action])
                self.current_action = 0
                self.action_duration = 0

        # Consider disabling renderer when not needed to improve speed:
        # self.pyboy.tick(1, False)
        self.pyboy.tick(1)

        lives = self.pyboy.memory[0xC17F]
        if lives == 0:  # != 5:??
            done = True
            print("Game Over")
        else:
            done = False

        self._calculate_fitness()
        reward = self._fitness - self._previous_fitness

        if self.policy == "MlpPolicy":
            observation = self.pyboy.game_area()
        else:
            observation = self.pyboy.screen.ndarray

        info = {}
        truncated = False

        return observation, reward, done, truncated, info

    def _calculate_fitness(self):
        self._previous_fitness = self._fitness

        # score the game
        self._fitness = self.calculate_reward()

    def calculate_reward(self):
        score_digits = self.pyboy.memory[0xC17A]
        lives = self.pyboy.memory[0xC17F]
        time_timer = self.pyboy.memory[0xC180]
        time_digits = self.pyboy.memory[0xC17D]
        time_reward = (time_timer + time_digits) / 2

        # Penalize low time reward
        penalty = 0
        if time_reward < 50:  # Example threshold
            penalty = 200  # Example penalty value

        # if player is just standing penalize
        if self.pyboy.memory[0xC1C2] == 0x00:
            penalty += 10

        return sum([score_digits * 5, lives + time_reward * 0.2]) - penalty

    def reset(self, **kwargs):
        # start with Level 1-1
        with open("level1-1.state", "rb") as f:
            self.pyboy.load_state(f)

        # self.pyboy.game_wrapper.reset_game()
        self._fitness = 0
        self._previous_fitness = 0
        self.current_action = 0
        self.action_duration = 0

        if self.policy == "MlpPolicy":
            observation = self.pyboy.game_area()
        else:
            observation = self.pyboy.screen.ndarray

        info = {}

        return observation, info

    def render(self, mode="human"):
        pass

    def close(self):
        self.pyboy.stop()
