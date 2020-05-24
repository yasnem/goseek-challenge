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
        self.last_aciton = -1

        self.grid = VoxelGrid(config['grid'])
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
        rgb, segmentation, depth, pose = decode_observations(observation)
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
