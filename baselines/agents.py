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
from src.agents.rrt import RRT

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
        self.rgb = None
        self.segmentation = None
        self.depth = None
        self.pose = None

    def reset(self) -> None:
        self.grid.reset()
        self.last_action = None
        self.rgb = None
        self.segmentation = None
        self.depth = None
        self.pose = None
        pass

    def _parse_observations(self, observations):
        rgb, segmentation, depth, pose = decode_observations(
            observations.reshape(1, -1))
        self.rgb = np.asarray(rgb[0, :, :, :]*255, dtype=np.uint8)
        self.segmentation = np.asarray(segmentation[0, :, :] * 10, dtype=np.uint8)
        self.depth = np.asarray(depth[0, :, :], dtype=np.float32)
        self.pose = np.asarray(pose[0, :], dtype=np.float32)

    def _process_observations(self):
        self.grid.integrate(rgb=self.rgb,
                            segmentation=self.segmentation,
                            depth=self.depth,
                            pose2d=self.pose)

    def act(self, observation: np.ndarray) -> int:
        self._parse_observations(observation)
        if self.last_action != 3 and self.pickup_agent.decide_pickup(self.segmentation, self.depth):
            action = 3
        else:
            self._process_observations()
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
        return 0 if self.grid.is_free_ahead(self.pose, 0.2, unobserved_as_free=True) else 1


def translate_action(action):
    if action == 0:
        return [0]
    elif action <= 22:
        # Repeat turning right multiple times.
        return [1]*action
    else:
        # Repeat turning left multiple times.
        return [2]*(45-action)


class NbvAgent(ShallowAgent):
    """ Next best view algorithm """
    def __init__(self, config):
        super().__init__(config)
        rrt_config = config['RRT']
        self.tree_depth = rrt_config['tree_depth']
        self.branch = rrt_config['branch']
        self.depth_limit = rrt_config['depth_limit']
        self.decay = rrt_config['decay']
        # This stores the current planned action.
        self.planned_actions = []

    def _plan(self):
        if not len(self.planned_actions):
            # Grow a tree at desired size.
            rrt = RRT(self.branch, self.tree_depth, self.grid, self.pose, self.decay, verbose=True)
            # Get reward and actions.
            found, best_reward, next_action = rrt.visit()
            while not found and rrt.level < self.depth_limit:
                rrt.grow()
                found, best_reward, next_action = rrt.visit()
            # Translate next action into executions.
            if not found:
                print("Failed to find best view. I become a bug.")
                self.planned_actions = [0] if self.grid.is_free_ahead(self.pose, 0.2, unobserved_as_free=True) else [1]
            else:
                self.planned_actions = translate_action(next_action)


        # Return an action.
        action = self.planned_actions.pop()
        return action