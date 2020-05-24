from typing import Any, Dict
import numpy as np
from tesse_gym.eval.agent import Agent
from stable_baselines3.sac import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common import logger
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.policies import FlattenExtractor
from agents.policy_utils import TSDFCNN, PickupAgent

import torch
from torch.utils.tensorboard import SummaryWriter


class SACAgent(Agent):
    """Stable Base SAC agent using also HER"""
    def __init__(self, env, log_dir, config: Dict[str, Any], weights=None) -> None:
        if weights is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if "cnn_arch" in config:
                self.model = SAC(policy=config['policy'],
                                 env=env,
                                 buffer_size=config['buffer_size'],
                                 gamma=config['gamma'],
                                 learning_rate=config['learning_rate'],
                                 batch_size=config['batch_size'],
                                 gradient_steps=config['gradient_steps'],
                                 policy_kwargs={'features_extractor_class':
                                                    TSDFCNN,
                                                'features_extractor_kwargs':
                                                    {'cnn_arch': config['cnn_arch'],
                                                     'features_dim': config['cnn_feature_dim']},
                                                'net_arch': config['net_arch']},
                                 device=self.device,
                                 verbose=config['verbose'])
            else:
                self.model = SAC(policy=config['policy'],
                                 env=env,
                                 buffer_size=config['buffer_size'],
                                 gamma=config['gamma'],
                                 learning_rate=config['learning_rate'],
                                 batch_size=config['batch_size'],
                                 gradient_steps=config['gradient_steps'],
                                 policy_kwargs={'features_extractor_class':
                                                    FlattenExtractor,
                                                'net_arch': config['net_arch']},
                                 device=self.device,
                                 verbose=config['verbose'])

            self.callback = CheckpointCallback(save_freq=config['save_freq'],
                                               save_path=log_dir)
            self.callback.init_callback(model=self.model)
            self.writer = SummaryWriter(log_dir+'/tb_log')
            self.log_dir = log_dir
            self.rollout_data = {'rewards': [[]],
                                 'ep_rewards': [],
                                 'ep_len': []}
            self.last_action = -1
        else:
            self.model = self.load(weights)

        if 'pickup' in config:
            self.pickup_agent = PickupAgent(config['pickup'], config[
                'voxel_grid']['camera'])
        else:
            self.pickup_agent = None

    def collect_rollouts(self,
                         env: VecEnv,
                         n_steps=-1,
                         learning_starts=0,
                         replay_buffer=None,
                         log_interval=None):
        """
        Collect rollout using the current policy and fill the replay buffer.

        :param env: (VecEnv) The training environment
        :param n_steps: (int) Number of steps to use to collect rollout data
            You can also specify a ``n_episodes`` instead.
        :param learning_starts: (int) Number of steps before learning for the warm-up phase.
        :param replay_buffer: (ReplayBuffer)
        :param log_interval: (int) Log data every ``log_interval`` episodes
        """
        total_steps = 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"

        while total_steps < n_steps:
            done = False

            while not done:

                # Select action randomly or according to policy
                if self.model.num_timesteps < learning_starts:
                    unscaled_action = np.array([self.model.action_space.sample()])
                else:
                    unscaled_action, _ = self.model.predict(self.model._last_obs,
                                                            deterministic=False)

                # Rescale the action from [low, high] to [-1, 1]
                scaled_action = self.model.policy.scale_action(unscaled_action)

                # Rescale and perform action
                new_obs, rewards, dones, infos = env.step(
                    self.model.policy.unscale_action(scaled_action))
                # episode_reward += reward
                self.callback.on_step()

                # Retrieve reward and episode length if using Monitor wrapper
                self.model._update_info_buffer(infos, dones)

                # Avoid changing the original ones
                self.model._last_original_obs, new_obs_, reward_ = \
                    self.model._last_obs, new_obs, rewards

                replay_buffer.add(self.model._last_original_obs, new_obs_,
                                  scaled_action, reward_, dones)
                done = dones[0]
                self.rollout_data['rewards'][-1].append(rewards[0])

                self.model._last_obs = new_obs

                self.model.num_timesteps += 1
                # episode_timesteps += 1
                total_steps += 1
                if 0 < n_steps <= total_steps:
                    break

            if done:
                self.model._episode_num += 1
                self.rollout_data['ep_len'].append(len(self.rollout_data['rewards'][-1]))
                self.rollout_data['ep_rewards'].append(np.mean(self.rollout_data['rewards'][-1]))
                self.rollout_data['rewards'].append([])

                ep_rewards_mean = np.mean(self.rollout_data['ep_rewards'])
                ep_len_mean = np.mean(self.rollout_data['ep_len'])
                last_ep_reward = np.mean(self.rollout_data['rewards'][-2])
                last_ep_len = len(self.rollout_data['rewards'][-2])

                self.writer.add_scalar('ep_reward_mean',
                                       ep_rewards_mean,
                                       self.model.num_timesteps)
                self.writer.add_scalar('ep_len_mean',
                                       ep_len_mean,
                                       self.model.num_timesteps)
                self.writer.add_scalar('last_ep_reward',
                                       last_ep_reward,
                                       self.model.num_timesteps)

                # Display training infos
                if self.model.verbose >= 1 and log_interval is not None and \
                        self.model._episode_num % log_interval == 0:
                    logger.logkv("episodes", self.model._episode_num)
                    if self.model._episode_num > 0:

                        logger.logkv('ep_rew_mean', ep_rewards_mean)
                        logger.logkv('ep_len_mean', ep_len_mean)
                        logger.logkv('last_ep_reward', last_ep_reward)
                        logger.logkv('last_ep_len', last_ep_len)
                    logger.dumpkvs()

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
        if self.pickup_agent is not None and self.last_action != 3:
            rgb, segmentation, depth = self.model.env.observe().images
            pickup_bool = self.pickup_agent.decide_pickup(segmentation, depth)
        else:
            pickup_bool = False
        observation = np.repeat(observation[np.newaxis], 1, 0)
        actions, state = self.model.predict(
            observation, state=self.state, deterministic=False
        )
        self.state = state  # update model state
        action = 3 if pickup_bool else int(actions[0])
        self.last_action = action
        return action

    def reset(self) -> None:
        """ Reset model state. """
        self.state = None

    def train(self, n_timesteps, log_interval):
        # Doesn't do much, and we don't want an eval env. So just fill with
        # default values from stable_baselines3.
        callback = self.model._setup_learn(None, None, -1, 5, None, True)
        callback.on_training_start(locals(), globals())
        n_episodes = 0
        self.model.rollout_data = {key: [] for key in ['observations',
                                                       'actions',
                                                       'rewards',
                                                       'dones',
                                                       'values']}
        while self.model.num_timesteps < n_timesteps:
            self.collect_rollouts(self.model.env,
                                  n_steps=self.model.train_freq,
                                  learning_starts=self.model.learning_starts,
                                  replay_buffer=self.model.replay_buffer,
                                  log_interval=log_interval)

            self.model._update_current_progress(self.model.num_timesteps,
                                                n_timesteps)

            if self.model.num_timesteps > 0 and self.model.num_timesteps > \
                    self.model.learning_starts:
                gradient_steps = self.model.gradient_steps
                self.model.train(gradient_steps,
                                 batch_size=self.model.batch_size)
                log_dict = logger.getkvs()
                self.writer.add_scalar('actor_loss',
                                       log_dict['actor_loss'],
                                       self.model.num_timesteps)
                self.writer.add_scalar('critic_loss',
                                       log_dict['critic_loss'],
                                       self.model.num_timesteps)
                self.writer.add_scalar('ent_coef',
                                       log_dict['ent_coef'],
                                       self.model.num_timesteps)

    def save(self):
        print("Training finished, saving model at ", self.log_dir + "/weights")
        self.model.save(self.log_dir + "/weights")

    @staticmethod
    def load(filename):
        print("Loading weight from ", filename)
        return SAC.load(filename)
