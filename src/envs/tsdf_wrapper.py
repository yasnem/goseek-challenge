import open3d as o3d
from gym import spaces
import numpy as np
from src.perception.tsdf_tools import VoxelGrid
from src.common.tools import extract_segment_class, get_rewards_config, \
    get_grid_config
from tesse_gym.tasks.goseek.goseek_full_perception import GoSeekFullPerception
from tesse_gym.tasks.navigation.navigation import Navigation
from src.envs.rewards import construct_reward
from tesse.msgs import DataResponse, RemoveObjectsRequest, ObjectsRequest
from typing import Callable, Dict, Tuple, Union, Optional
from collections import deque
from tesse_gym.core.tesse_gym import TesseGym
from tesse_gym.core.utils import NetworkConfig, set_all_camera_params

import defusedxml.ElementTree as ET
import math


class TsdfBase():
    def __init__(self, rewards, grid):
        self.grid = VoxelGrid(grid)
        self.compute_target = self.grid.compute_target
        self.obs_shape = (self.grid.channels, self.grid.grid_size[0],
                          self.grid.grid_size[1], self.grid.grid_size[2])
        if self.compute_target:
            self.obs_shape = (self.grid.channels, self.grid.grid_size[0] + 1,
                          self.grid.grid_size[1], self.grid.grid_size[2])
        self.debug = grid['debug']
        self.rewards = [construct_reward(reward) for reward in rewards]
        self.observed_q = deque()
        self.current_target = None
        self.target_relative = None

    def get_target_relative(self, target, pose2d):
        """ Get the target pose by frontier computation. """
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
        return np.asarray([target[0]-pose2d[0], target[1]-pose2d[1], relative_orientation])

    def clear(self):
        self.grid.reset()
        self.observed_q = deque()
        self.current_target = None
        self.target_relative = None


class GoseekTSDF(GoSeekFullPerception, TsdfBase):
    def __init__(
            self,
            build_path: str,
            network_config: Optional[NetworkConfig] = NetworkConfig(),
            scene_id: Optional[int] = None,
            episode_length: Optional[int] = 400,
            step_rate: Optional[int] = 20,
            n_targets: Optional[int] = 30,
            success_dist: Optional[float] = 2,
            restart_on_collision: Optional[bool] = False,
            auto_pickup: Optional[bool] = False,
            init_hook: Optional[Callable[[TesseGym], None]] = set_all_camera_params,
            rewards: Optional[list] = get_rewards_config('goseek'),
            ground_truth_mode: Optional[bool] = True,
            n_target_types: Optional[int] = 5,
            grid:  Optional[list] = get_grid_config('goseek')):
        GoSeekFullPerception.__init__(self, build_path=build_path, network_config=network_config, scene_id=scene_id,
                         episode_length=episode_length, step_rate=step_rate, n_targets=n_targets,
                         success_dist=success_dist, restart_on_collision=restart_on_collision,
                         init_hook=init_hook, ground_truth_mode=ground_truth_mode,
                         n_target_types=n_target_types)
        TsdfBase.__init__(self, rewards=rewards, grid=grid)
        self.auto_pickup = auto_pickup

    @property
    def action_space(self) -> spaces.Box:
        """ Actions available to agent. """
        if not self.auto_pickup: # original training.
            return spaces.Box(np.array([0]), np.array([4 - 1e-6]),dtype=np.float32)
        else: # Navigation only.
            return spaces.Box(np.array([0]), np.array([3 - 1e-6]),dtype=np.float32)

    @property
    def observation_space(self) -> spaces.Box:
        """ Define an observation space for RGB, depth, segmentation, and pose.
        Because Stables Baselines (the baseline PPO library) does not support dictionary spaces,
        the observation images and pose vector will be combined into a vector.
        """
        return spaces.Box(-1.0, 1.0, shape=self.obs_shape)

    # Overloading this modifies the observation from simulator.
    def form_agent_observation(self, response: DataResponse) -> np.ndarray:
        rgb, segmentation, depth = response.images
        pose = self.get_pose()
        self.grid.integrate(rgb=rgb,
                            segmentation=extract_segment_class(segmentation),
                            depth=depth,
                            pose2d=pose)
        self.grid.observe_local(pose)

        # Record current number of observation and remove values older than 1 frame.
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

    def reset(self, **kwargs):
        TsdfBase.clear(self)
        return GoSeekFullPerception.reset(self)

    # Overload function from goseek.
    def compute_reward(
            self, observation: DataResponse, action: int
    ) -> Tuple[float, Dict[str, Union[int, bool]]]:

        reward_info = {"env_changed": False, "collision": False, "n_found_targets": 0}
        # Compute currently visible targets.
        found_targets = []
        if action == 3 or self.auto_pickup:
            # update the environment if pickup action is selected or auto pickup is used.
            found_targets = self.collect_visible_targets(observation)
            if len(found_targets) > 0:
                self.n_found_targets += len(found_targets)
                self.env.request(RemoveObjectsRequest(ids=found_targets))
                reward_info["env_changed"] = True
                reward_info["n_found_targets"] += len(found_targets)

            if self.auto_pickup: # not really found, so remove found target.
                found_targets = []
            # if all targets have been found, restart the episode
            if self.n_found_targets == self.n_targets:
                self.done = True

        # Compute reward from the reward structure.
        reward = 0
        newly_discovered = 0 if len(self.observed_q) < 2 else self.observed_q[1] - self.observed_q[0]
        cur_diff = self.get_target_relative(self.current_target, self.get_pose())
        dist_diff = np.linalg.norm(cur_diff[:2]) - np.linalg.norm(self.target_relative[:2])
        angular_diff = abs(cur_diff[2]) - abs(self.target_relative[2])
        assert newly_discovered >= 0
        for r in self.rewards:
            cur_r = r(observation, action, self.get_pose(), found_targets, newly_discovered, [dist_diff, angular_diff])
            reward += cur_r
            if self.debug:
                print(r.type + ": " + str(cur_r))

        self.steps += 1
        if self.steps > self.episode_length:
            self.done = True

        # collision information isn't provided by the controller metadata
        if self._collision(observation.metadata):
            reward_info["collision"] = True

            if self.restart_on_collision:
                self.done = True
        return reward, reward_info

    def collect_visible_targets(self, observation):
        targets = self.env.request(ObjectsRequest())

        # If not in ground truth mode, metadata will only provide position estimates
        # In that case, get ground truth metadata from the controller
        agent_metadata = (
            observation.metadata
            if self.ground_truth_mode
            else self.continuous_controller.get_broadcast_metadata()
        )
        # compute agent's distance from targets
        agent_position = self._get_agent_position(agent_metadata)
        target_ids, target_position = self._get_target_id_and_positions(
            targets.metadata
        )
        if target_position.shape[0] == 0:
            return []
        else:
            return self.get_found_targets(
                agent_position, target_position, target_ids, agent_metadata)


class NavigationTSDF(Navigation, TsdfBase):
    def __init__(self,
            build_path: str,
            network_config: Optional[NetworkConfig] = NetworkConfig(),
            scene_id: Optional[int] = None,
            episode_length: Optional[int] = 400,
            step_rate: Optional[int] = -1,
            init_hook: Optional[Callable[[TesseGym], None]] = set_all_camera_params,
            rewards: Optional[list] = get_rewards_config('navigation'),
            ground_truth_mode: Optional[bool] = True,
            grid:  Optional[list] = get_grid_config('navigation')):
        Navigation.__init__(self, sim_path=build_path, network_config=network_config, scene_id=scene_id,
                         episode_length=episode_length, step_rate=step_rate,
                         init_hook=init_hook, ground_truth_mode=ground_truth_mode)

        TsdfBase.__init__(self, rewards=rewards, grid=grid)

    def compute_reward(self, observation, action):
        # Compute reward from the reward structure.
        reward = 0
        newly_discovered = 0 if len(self.observed_q) < 2 else self.observed_q[1] - self.observed_q[0]
        cur_diff = self.get_target_relative(self.current_target, self.get_pose())
        dist_diff = np.linalg.norm(cur_diff[:2]) - np.linalg.norm(self.target_relative[:2])
        angular_diff = abs(cur_diff[2]) - abs(self.target_relative[2])
        assert newly_discovered >= 0
        for r in self.rewards:
            cur_r = r(observation, action, self.get_pose(), [], newly_discovered, [dist_diff, angular_diff])
            reward += cur_r
            if self.debug:
                print(r.type + ": " + str(cur_r))
        self.steps += 1
        collided = ET.fromstring(observation.metadata).find("collision").attrib["status"].lower() == "true"
        reward_info = {"env_changed": False, "collision": collided, "n_found_targets": 0}
        self.done = collided or self.steps > self.episode_length

        if self.debug:
            print(reward_info)
        return reward, reward_info

    @property
    def observation_space(self) -> spaces.Box:
        """ Define an observation space for RGB, depth, segmentation, and pose.
        Because Stables Baselines (the baseline PPO library) does not support dictionary spaces,
        the observation images and pose vector will be combined into a vector.
        """
        return spaces.Box(-1.0, 1.0, shape=self.obs_shape)


    # Overloading this modifies the observation from simulator.
    def form_agent_observation(self, response: DataResponse) -> np.ndarray:
        rgb, segmentation, depth = response.images
        pose = self.get_pose()
        self.grid.integrate(rgb=rgb,
                            segmentation=extract_segment_class(segmentation),
                            depth=depth,
                            pose2d=pose)
        self.grid.observe_local(pose)

        # Record current number of observation and remove values older than 1 frame.
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
            pose_mask[0, 0, :3, 0] = self.grid.get_target_position(pose)
            feature = np.concatenate([feature, pose_mask], axis=1)

        return feature

    def reset(self, **kwargs):
        TsdfBase.clear(self)
        return Navigation.reset(self)
