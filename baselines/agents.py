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
import random
from tesse_gym.eval.agent import Agent
from tesse_gym.tasks.goseek.goseek_full_perception import decode_observations
from src.agents.policy_utils import PickupAgent, get_target_relative
from src.perception.tsdf_tools import VoxelGrid


class ShallowAgent(Agent):
    """ Agent that goes straight when it can. """

    def __init__(self, config: Dict[str, Any]) -> None:
        """ Initialize agent.

        Args:
            config (Dict[str, Any]): Agent configuration
        """
        self.pickup_agent = PickupAgent(config['pickup'],
                                        config['voxel_grid']['camera'])
        self.target_class = 10
        self.last_action = None
        self.grid = VoxelGrid(config['voxel_grid'])

    def reset(self) -> None:
        self.grid.reset()
        self.last_action = None
        pass

    def _parse_observations(self, observations):
        rgb, segmentation, depth, pose = decode_observations(
            observations.reshape(1, -1))
        rgb = np.asarray(rgb[0, :, :, :]*255, dtype=np.uint8)
        segmentation = np.asarray(segmentation[0, :, :] * 10, dtype=np.uint8)
        depth = np.asarray(depth[0, :, :], dtype=np.float32)
        pose = np.asarray(pose[0, :], dtype=np.float32)
        return rgb, segmentation, depth, pose

    def _process_observations(self, rgb, segmentation, depth, pose):
        self.grid.integrate(rgb=rgb,
                            segmentation=segmentation,
                            depth=depth,
                            pose2d=pose)
        self.grid.observe_local(pose)
        return rgb, segmentation, depth, pose

    def act(self, observation: np.ndarray) -> int:
        rgb, segmentation, depth, pose = self._parse_observations(observation)
        if self.last_action != 3 and self.pickup_agent.decide_pickup(segmentation, depth):
            action = 3
        else:
            self._process_observations(rgb, segmentation, depth, pose)
            # current_target = self.grid.get_target(pose)
            # rel_target = get_target_relative(current_target, pose)
            # Go straight unless occupied.
            action = self._plan()

        self.last_action = action
        return action

    def _plan(self):
        raise NotImplementedError("Extend this method.")


class BugAgent(ShallowAgent):
    """ Agent with bug algorithm, keeps going straight until it can't """
    def __init__(self, config):
        super().__init__(config)

    def _plan(self):
        return 0 if self.grid.is_free_ahead(0.2, unobserved_as_free=True) else 0
