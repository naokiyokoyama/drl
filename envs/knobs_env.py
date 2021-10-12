import gym
from gym import spaces
import numpy as np


class KnobsEnv(gym.Env):
    """Agent has to move knobs to goal positions!"""

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = {}
        self.max_movement = np.deg2rad(config.get('MAX_MOVEMENT', 3))
        self.num_knobs = config.get('NUM_KNOBS', 3)
        self.success_thresh = config.get('SUCCESS_THRESH', 3)
        self.max_steps = config.get('MAX_STEPS', 200)

        self.current_state = None
        self.goal_state = None
        self.num_steps = 0
        self.error = None
        self.cumul_reward = 0

        # Agent has an action for each knob, i.e. how much each knob is changed by
        self.action_space = spaces.Box(
            low=-self.max_movement, high=self.max_movement,
            shape=(self.num_knobs,), dtype=np.float32
        )

        # Agent is given the current angle of each knob (-180 to 180)
        self.observation_space = spaces.Box(
            low=-np.pi, high=np.pi,
            shape=(self.num_knobs,), dtype=np.float32
        )

    def reset(self, goal_state=None):
        # Randomize goal state if not specified
        if goal_state is None:
            goal_state = self._get_random_knobs()

        self.goal_state = goal_state
        self.current_state = self._get_random_knobs()
        self.num_steps = 0
        observations = self.error = self.goal_state - self.current_state
        self.cumul_reward = 0

        return observations

    def _get_random_knobs(self):
        # (0, 1) -> (0, 2) -> (-1, 1) -> (-np.pi, np.pi)
        return (
            np.random.rand(self.num_knobs) * 2 - 1
        ) * np.pi

    def step(self, action):
        action = np.clip(action, -self.max_movement, self.max_movement)
        self.current_state = np.clip(self.current_state + action, -np.pi, np.pi)

        # Get error between current and goal states
        error = np.abs(self.goal_state - self.current_state)
        reward = np.mean([
            (pe - e)**2
            for pe, e in zip(self.error, error)
        ]) / 2.0

        self.num_steps += 1
        self.error = error
        success = all([
            abs(c - g) < np.deg2rad(5)
            for c, g in zip(self.current_state, self.goal_state)
        ])

        observations = self.goal_state - self.current_state
        if success:
            reward += 10
        done = success or self.num_steps == self.max_steps

        self.cumul_reward += reward
        info = {
            'error': error,
            'reward': reward,
            'success': success,
            'failed': self.num_steps == self.max_steps,
            'cumul_reward': self.cumul_reward,
            'episode': {
                'r': reward
            }
        }
        # print(*[
        #     np.rad2deg(i[0]) for i in (action, self.current_state, self.goal_state)
        # ], reward)

        return observations, reward, done, info

    def _validate_heading(self, heading):
        if heading > np.pi:
            heading -= np.pi * 2
        elif heading < -np.pi:
            heading += np.pi * 2

        return heading

    def _get_heading_error(self, source, target):
        return self._validate_heading(target - source)