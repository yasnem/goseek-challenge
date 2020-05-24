from typing import Any, Dict
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
from .policy_utils import image_and_pose_network
from tesse_gym.eval.agent import Agent


class PPOAgent(Agent):
    def __init__(self, env, log_dir, n_envs, config: Dict[str, Any]) -> None:
        self.logdir = log_dir
        self.model = PPO(
            policy=config['policy'],
            env=env,
            verbose=1,
            tensorboard_log=self.logdir + "/tensorboard/",
            nminibatches=config['number_mini_batches'] * n_envs,
            gamma=config['gamma'],
            learning_rate=config['learning_rate'],
            policy_kwargs={'cnn_extractor': image_and_pose_network})

        self.n_train_envs = self.model.initial_state.shape[0]
        assert self.n_train_envs == n_envs
        self.state = None
        self.callback = CheckpointCallback(save_freq=config['save_freq'],
                                           save_path=self.logdir)

    def act(self, observation: np.ndarray) -> int:
        """ Act on an observation.
        args:
            observation (np.ndarray): observation.
        returns:
            int: an action in [0, 4) defined as follows
                - 0: forward 0.5m
                - 1: right 8 degrees
                - 2: left 8 degrees
                - 3: declare target
        """
        observation = np.repeat(observation[np.newaxis], self.n_train_envs, 0)
        actions, state = self.model.predict(
            observation, state=self.state, deterministic=False
        )
        self.state = state  # update model state
        return actions[0]

    def reset(self) -> None:
        """ Reset model state. """
        self.state = None

    def get_env(self):
        return self.model.get_env()

    def train_envs(self):
        return self.model.act_model.initial_state.shape[0]

    def train(self, steps):
        self.model.learn(total_timesteps=steps, callback=self.callback)

    def save(self):
        print("Training finished, saving model at ", self.logdir + "/weights")
        self.model.save(self.logdir + "/weights")

    def load(self, filename):
        print("Loading weight from ", filename)
        self.model = PPO.load(filename)