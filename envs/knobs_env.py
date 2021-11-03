import gym
from gym import spaces
import numpy as np


class KnobsEnv(gym.Env):
    """Agent has to move knobs to goal positions!"""

    def __init__(self, config=None, seed=1):
        super().__init__()
        if config is None:
            config = {}

        np.random.seed(seed)

        self.max_movement = np.deg2rad(config.MAX_MOVEMENT)
        self.num_knobs = config.NUM_KNOBS
        self.success_thresh = np.deg2rad(config.SUCCESS_THRESH)
        self._max_episode_steps = config.MAX_STEPS

        self.current_state = None
        self.goal_state = None
        self.num_steps = 0
        self.cumul_reward = 0

        # Agent has an action for each knob, i.e. how much each knob is changed by
        self.action_space = spaces.Box(
            low=-self.max_movement, high=self.max_movement,
            shape=(self.num_knobs,), dtype=np.float32
        )

        # Agent is given the current angle difference of each knob (-180 to 180)
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
        observations = self.error = self.get_observations()
        self.cumul_reward = 0

        return observations

    def get_observations(self):
        return np.array([
            self._get_heading_error(c, g)
            for c, g in zip(self.current_state, self.goal_state)
        ])

    def _get_random_knobs(self):
        # (0, 1) -> (0, 2) -> (-1, 1) -> (-np.pi, np.pi)
        return (
            np.random.rand(self.num_knobs) * 2 - 1
        ) * np.pi

    def step(self, action):
        # Clip actions and scale
        action = np.clip(action, -1.0, 1.0) * self.max_movement

        # Update current state
        self.current_state = np.array([
            self._validate_heading(c + a)
            for c, a in zip(self.current_state, action)
        ])

        # Return observations (error for each knob)
        observations = self.get_observations()

        # Penalize MSE between current and goal states
        reward_terms = -observations ** 2 / self.num_knobs
        reward = sum(reward_terms)

        # Check termination conditions
        success = all(
            [abs(i) < self.success_thresh for i in observations]
        )

        self.num_steps += 1
        done = success or self.num_steps == self._max_episode_steps

        self.cumul_reward += reward
        info = {
            'reward': reward,
            'success': success,
            'failed': self.num_steps == self._max_episode_steps,
            'cumul_reward': self.cumul_reward,
            'reward_terms': reward_terms,
            'episode': {
                'r': reward
            }
        }

        return observations, reward, done, info

    def _validate_heading(self, heading):
        if heading > np.pi:
            heading -= np.pi * 2
        elif heading < -np.pi:
            heading += np.pi * 2

        return heading

    def _get_heading_error(self, source, target):
        return self._validate_heading(target - source)