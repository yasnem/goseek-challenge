###################################################################################################
# DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.
#
# This material is based upon work supported by the Under Secretary of Defense for Research and
# Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions, findings, conclusions
# or recommendations expressed in this material are those of the author(s) and do not necessarily
# reflect the views of the Under Secretary of Defense for Research and Engineering.
#
# (c) 2020 Massachusetts Institute of Technology.
#
# MIT Proprietary, Subject to FAR52.227-11 Patent Rights - Ownership by the contractor (May 2014)
#
# The software/firmware is provided to you on an As-Is basis
#
# Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part 252.227-7013
# or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government rights in this work
# are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed above. Use of this work other
# than as specifically authorized by the U.S. Government may violate any copyrights that exist in
# this work.
###################################################################################################

from typing import Any, Dict

import numpy as np
import math

from tesse_gym.eval.agent import Agent
from tesse_gym.tasks.goseek.goseek_full_perception import decode_observations
from src.agents.policy_utils import PickupAgent
from src.perception.tsdf_tools import VoxelGrid
from collections import deque


class ShallowAgent(Agent):
    """ Agent that takes random actions. """

    def __init__(self, config: Dict[str, Any]) -> None:
        """ Initialize agent.

        Args:
            config (Dict[str, Any]): Agent configuration
        """
        self.action_space = np.arange(0, 4)

        self.pickup_agent = PickupAgent(config['pickup'],
                                        config['voxel_grid']['camera'])
        self.target_class = 10
        self.last_action = -1

        self.grid = VoxelGrid(config['voxel_grid'])
        self.observed_q = deque()
        self.current_target = None
        self.target_relative = None
        self.compute_target = self.grid.compute_target
        self.obs_shape = (self.grid.channels, self.grid.grid_size[0],
                          self.grid.grid_size[1], self.grid.grid_size[2])
        if self.compute_target:
            self.obs_shape = (self.grid.channels, self.grid.grid_size[0] + 1,
                              self.grid.grid_size[1], self.grid.grid_size[2])

    def act(self, observation: np.ndarray) -> int:
        """ Take a uniformly random action.

        args:
            observation (np.ndarray): observation.

        returns:
            int: an action in [0, 4) defined as follows
                - 0: forward 0.5m
                - 1: right 8 degrees
                - 2: left 8 degrees
                - 3: declare target
        """
        rgb, segmentation, depth, pose = decode_observations(
            observation.reshape(1, -1))
        rgb = np.asarray(rgb[0, :, :, :]*255, dtype=np.uint8)
        segmentation = np.asarray(segmentation[0, :, :] * 10, dtype=np.uint8)
        depth = np.asarray(depth[0, :, :], dtype=np.float32)
        pose = np.asarray(pose[0, :], dtype=np.float32)

        tsdf_features = self.form_agent_observation(rgb=rgb,
                                                    segmentation=segmentation,
                                                    depth=depth,
                                                    pose=pose)
        if self.last_action != 3:
            pickup_bool = self.pickup_agent.decide_pickup(segmentation, depth)
        else:
            pickup_bool = False
        # TODO(Y): Add proper action selection using JDS algorithm
        action = 0

        action = 3 if pickup_bool else action
        return action

    def get_target_relative(self, target, pose2d):
        """Get the target pose by frontier computation."""
        assert self.compute_target
        # Function signature of get target is pose3d and yaw orientation.
        # TODO(jd): Put this in range of -1 till 1.
        # make sure relative orientation is within [-pi, pi]
        relative_orientation = target[2] - pose2d[2]
        if relative_orientation > math.pi:
            relative_orientation -= 2*math.pi
        if relative_orientation < -math.pi:
            relative_orientation += 2 * math.pi
        """ The principal component direction is currently unused. """
        return np.asarray([target[0]-pose2d[0],
                           target[1]-pose2d[1],
                           relative_orientation])

    def form_agent_observation(self, rgb, segmentation, depth, pose):

        self.grid.integrate(rgb=rgb,
                            segmentation=segmentation,
                            depth=depth,
                            pose2d=pose)
        self.grid.observe_local(pose)

        # Record current number of observation.
        # Remove values older than 1 frame.
        self.observed_q.append(self.grid.get_observed_voxels())
        if len(self.observed_q) > 2:
            self.observed_q.popleft()

        tsdf = self.grid.get_tsdf()

        if self.grid.channels == 1:
            feature = np.expand_dims(tsdf, axis=0)
        elif self.grid.channels == 2:
            seg = self.grid.get_target_likelihood()
            feature = np.stack([tsdf, seg])
        else:
            raise NotImplementedError("channel number can only be 1 or 2. ")

        if self.compute_target:
            pose_mask = np.zeros((self.grid.channels, 1, feature.shape[2], feature.shape[3]))
            self.current_target = self.grid.get_target(pose)
            self.target_relative = self.get_target_relative(self.current_target, pose)
            pose_mask[0, 0, :3, 0] = self.target_relative
            feature = np.concatenate([feature, pose_mask], axis=1)

        return feature

    def reset(self) -> None:
        """ Nothing required on episode reset. """
        self.grid.reset()
        self.observed_q = deque()
        self.current_target = None
        self.target_relative = None
        pass



class StableBaselinesPPO(Agent):
    """ Stable Baselines PPO agent for GOSEEK submission. """

    def __init__(self, config: Dict[str, Any]) -> None:
        """ Initialize agent.

        Args:
            config (Dict[str, Any]): Agent configuration.
        """
        from stable_baselines import PPO2
        self.model = PPO2.load(config["weights"])
        self.state = None

        # Number of environments used to train model
        # to which stable-baselines input tensor size is fixed
        self.n_train_envs = self.model.initial_state.shape[0]

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


class RandomAgent(Agent):
    """ Agent that takes random actions. """

    def __init__(self, config: Dict[str, Any]) -> None:
        """ Initialize agent.

        Args:
            config (Dict[str, Any]): Agent configuration
        """

        self.action_space = np.arange(0, 4)

        # give probability for actions in `self.action_space`
        self.action_probability = np.array(config["action_probability"])
        self.action_probability /= self.action_probability.sum()

    def act(self, observation: np.ndarray) -> int:
        """ Take a uniformly random action.

        args:
            observation (np.ndarray): observation.

        returns:
            int: an action in [0, 4) defined as follows
                - 0: forward 0.5m
                - 1: right 8 degrees
                - 2: left 8 degrees
                - 3: declare target
        """
        return np.random.choice(self.action_space, p=self.action_probability)

    def reset(self) -> None:
        """ Nothing required on episode reset. """
        pass
