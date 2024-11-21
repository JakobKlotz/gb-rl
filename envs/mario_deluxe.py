# Adopted from https://github.com/Baekalfen/PyBoy/wiki/Using-PyBoy-with-Gym
# Memory addresses: https://datacrystal.tcrf.net/wiki/Super_Mario_Bros._Deluxe/RAM_map
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MarioDeluxe(gym.Env):
    def __init__(
        self,
        pyboy,
        policy: str = "MlpPolicy",
        debug=False,
        render: bool = True,
        n_frames: int = 5,
    ):
        super().__init__()

        if policy not in ("MlpPolicy", "CnnPolicy"):
            raise ValueError("Policy must be 'MlpPolicy' or 'CnnPolicy'")

        self.pyboy = pyboy
        self.render = render
        self.n_frames = n_frames
        self.policy = policy
        self.matrix_shape = (
            (32, 32, 1) if policy == "MlpPolicy" else (144, 160, 3)
        )
        self.actions = [
            ("",),
            ("right",),
            ("right", "a"),
            ("right", "b"),
            ("right", "a", "b"),
            ("a",),
            ("left",),
        ]
        self._fitness = 0
        self._previous_fitness = 0
        self.debug = debug
        self.previous_action = 0

        # keep track of player's x position
        self.last_x_pos = None

        if not self.debug:
            self.pyboy.set_emulation_speed(0)

        self.action_space = spaces.Discrete(len(self.actions))

        # MlpPolicy uses a convenience wrapper from PyBoy to get the game area
        # the game are (the observation) is normalized from 0 to 1
        if self.policy == "MlpPolicy":
            low, high = 0, 1
            dtype = np.float32
        # use pixel values for CnnPolicy as a GreyScaling is applied afterward
        else:
            low, high = 0, 255
            dtype = np.uint8
        self.observation_space = spaces.Box(
            low=low, high=high, shape=self.matrix_shape, dtype=dtype
        )

        self.pyboy.game_wrapper.start_game()

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (
            action,
            type(action),
        )

        # Move the agent
        if self.previous_action != 0:
            previous_buttons = self.actions[self.previous_action]
            # release buttons which were in previous action
            for button in previous_buttons:
                self.pyboy.button_release(button)
        if action != 0:
            # press buttons for current action
            current_buttons = self.actions[action]
            for button in current_buttons:
                self.pyboy.button_press(button)

        # save
        self.previous_action = action

        self.pyboy.tick(count=self.n_frames, render=self.render)

        # done; if player is dead
        done = self.pyboy.memory[0xC1C1] == 3

        self._calculate_fitness()
        reward = self._fitness - self._previous_fitness

        if self.policy == "MlpPolicy":
            observation = (
                self.pyboy.game_area() / 378
            )  # 378 is apparently the max value
            observation = observation.reshape((32, 32, 1))
        else:
            observation = self.pyboy.screen.ndarray[:, :, :3]

        info = {}
        truncated = False

        return observation, reward, done, truncated, info

    def _calculate_fitness(self):
        self._previous_fitness = self._fitness

        # score the game
        self._fitness = self.calculate_reward()

    def calculate_reward(self):
        # get player x position
        x_one, x_two = self.pyboy.memory[0xC1CA], self.pyboy.memory[0xC1CB]
        player_x = x_one + x_two * 256

        # get score (has size 3)
        digit_one, digit_two, digit_three = (
            self.pyboy.memory[0xC17A],
            self.pyboy.memory[0xC17B],
            self.pyboy.memory[0xC17C],
        )
        score = digit_one * 10 + digit_two * 256 * 10 + digit_three
        # score ranges in the hundreds to thousands
        score /= 2_000

        powerup_state = self.pyboy.memory[
            0xC1C5
        ]  # 0: small, 1: big, 2+: powered up
        player_state = self.pyboy.memory[0xC1C1]  # 3: dead
        player_pose = self.pyboy.memory[0xC1C2]

        # Initialize reward components
        progress_reward = 0
        state_reward = 0
        time_penalty = 0
        death_penalty = 0

        # Calculate progress reward
        if self.last_x_pos is not None:
            progress_reward = (
                player_x - self.last_x_pos
            ) * 2.0  # Reward forward movement

        self.last_x_pos = player_x

        # level completion reward
        flag_reached = player_pose == 12
        flag_reward = 25 if flag_reached else 0

        # State-based rewards
        state_reward += powerup_state  # Reward power-up state;
        # with simply its value (0, 1, 2) as reward

        # Time management (time is stored in two bytes)
        time_high, time_low = (
            self.pyboy.memory[0xC17D],
            self.pyboy.memory[0xC17E],
        )
        timer = time_low * 256 + time_high  # -> time low is either 1 or 0

        # TODO implement a more sophisticated time penalty
        if timer < 50:  # Low on time
            time_penalty = -10

        # Penalty for death
        if player_state == 3:  # Dead state
            death_penalty = -25

        # Movement rewards/penalties
        movement_reward = 0
        if player_pose == 0:  # Standing still
            movement_reward = -5
        elif player_pose in [1, 2, 6]:  # Walking poses
            movement_reward = 1
        elif player_pose == 4:  # Jumping
            movement_reward = 1.5

        # Calculate final reward
        total_reward = (
            score
            + progress_reward
            + state_reward
            + flag_reward
            + movement_reward
            + time_penalty
            + death_penalty
        )

        # Clip reward to prevent extreme values
        return np.clip(total_reward, -25, 25)

    def reset(self):
        # start with Level 1-1
        with Path("state/level1-1.state").open("rb") as f:
            self.pyboy.load_state(f)

        self.previous_action = 0
        self._fitness = 0
        self._previous_fitness = 0
        self.last_x_pos = 0

        if self.policy == "MlpPolicy":
            observation = self.pyboy.game_area() / 378  # ranges from 0 to 378
            observation = observation.reshape((32, 32, 1))
        else:
            observation = self.pyboy.screen.ndarray[:, :, :3]

        info = {}

        return observation, info

    def render(self, mode="human"):
        pass

    def close(self):
        self.pyboy.stop()
