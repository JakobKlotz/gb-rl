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
            "right",
        ]
        self._fitness = 0
        self._previous_fitness = 0
        self.debug = debug

        # to 'support' continuous button presses and
        # simultaneous button presses
        self.current_action = 0  # no action ""
        self.action_duration = 0
        self.max_action_duration = 4  # number of steps to hold a button
        # currently, 30 frames per step = 4*30 = 120 frames -> 2 seconds

        # keep track of player's x position
        self.last_x_pos = 0

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
            self.current_action = 0
            self.action_duration = 0
        elif action != self.current_action:
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
        self.pyboy.tick(30)

        is_dead = self.pyboy.memory[0xC1C1] == 3
        if is_dead:
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
        # get player x position
        x_one, x_two = self.pyboy.memory[0xC1CA], self.pyboy.memory[0xC1CB]
        player_x = x_one + x_two * 256

        # get score (has size 3)
        digit_one, digit_two, digit_three = self.pyboy.memory[0xC17A], self.pyboy.memory[0xC17B], self.pyboy.memory[0xC17C]
        score = digit_one * 10 + digit_two * 256 * 10 + digit_three

        # lives = self.pyboy.memory[0xC17F]
        coins = self.pyboy.memory[0xC1F2]
        powerup_state = self.pyboy.memory[0xC1C5]  # 0: small, 1: big, 2+: powered up
        player_state = self.pyboy.memory[0xC1C1]  # 3: dead
        player_pose = self.pyboy.memory[0xC1C2]

        # Initialize reward components
        progress_reward = 0
        state_reward = 0
        time_penalty = 0
        death_penalty = 0

        # Calculate progress reward
        if hasattr(self, 'last_x_pos'):
            progress_reward = (player_x - self.last_x_pos) * 0.5  # Reward forward movement
        self.last_x_pos = player_x

        # level completion reward
        flag_reached = player_pose == 12
        flag_reward = 2000 if flag_reached else 0

        # State-based rewards
        state_reward += coins * 10  # Reward coin collection
        state_reward += powerup_state * 50  # Reward power-up state

        # Time management (time is stored in two bytes)
        time_high, time_low = self.pyboy.memory[0xC17D], self.pyboy.memory[0xC17E]
        timer = time_low * 256 + time_high  # -> time low is either 1 or 0
        # time_timer = self.pyboy.memory[0xC180]
        if timer < 50:  # Low on time
            time_penalty = -10

        # Penalty for death
        if player_state == 3:  # Dead state
            death_penalty = -500

        # Movement rewards/penalties
        movement_reward = 0
        if player_pose == 0:  # Standing still
            movement_reward = -5
        elif player_pose in [1, 2, 6]:  # Walking poses
            movement_reward = 2
        elif player_pose == 4:  # Jumping
            movement_reward = 3

        # Calculate final reward
        total_reward = (
                score / 100 +
                progress_reward +
                state_reward +
                flag_reward +
                movement_reward +
                time_penalty +
                death_penalty
        )

        # Clip reward to prevent extreme values
        total_reward = max(min(total_reward, 2000), -1000)

        return total_reward

    def reset(self, **kwargs):
        # start with Level 1-1
        with open("level1-1.state", "rb") as f:
            self.pyboy.load_state(f)

        # self.pyboy.game_wrapper.reset_game()
        self._fitness = 0
        self._previous_fitness = 0
        self.current_action = 0
        self.action_duration = 0
        self.last_x_pos = 0

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
