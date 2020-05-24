import numpy as np

import defusedxml.ElementTree as ET

class RewardBase(object):
    def __init__(self, weight):
        self._weight = weight

    def __call__(self, observation_after, action, relative_pose) -> float:
        """ call the function to compute the reward with the given observation and action. the observation comes after the action.

        Args:
            observation_after: observation after the action is taken.
            action (object): an action executed by the agent
            relative_pose: pose relative to starting point
        Returns:
            reward (float) : amount of reward returned from this reward previous action
        """
        raise NotImplementedError

class RewardByTargetPursuit(RewardBase):
    """ Reward if the agent has tried to follow the frontier. """
    def __init__(self, weight, weight_angular):
        super().__init__(weight)
        self.type = "pursuit"
        self._weight_angular = weight_angular

    """delta_to_target is the difference of the difference to relative target. Defined as #Diff_To_Target_Now - 
    #Diff_To_Target_Before, first element for distance, second element for angular."""
    def __call__(self, observation_after, action, relative_pose, found_targets, new_voxels, delta_to_target):
        if delta_to_target[0] < 0:
            return self._weight
        elif delta_to_target[1] < 0:
            return self._weight_angular
        return 0

class RewardByDecay(RewardBase):
    """ Decay Loss """
    def __init__(self, weight):
        super().__init__(weight)
        self.type = "decay"
    def __call__(self, observation_after, action, relative_pose, found_targets, new_voxels, delta_to_target):
        return self._weight


class RewardByForwardMovement(RewardBase):
    """ Movement reward """
    def __init__(self, weight):
        super().__init__(weight)
        self.type = "forward"
    def __call__(self, observation_after, action, relative_pose, found_targets, new_voxels, delta_to_target):
        return self._weight if action == 0 else 0


class RewardByDistanceTraveled(RewardBase):
    """Reward by distance from start """
    def __init__(self, weight):
        super().__init__(weight)
        self.type = "travelled total"
    def __call__(self, observation_after, action, relative_pose, found_targets, new_voxels, delta_to_target):
        return self._weight * np.linalg.norm(relative_pose[:2])


class RewardByCollision(RewardBase):
    """Reward if collided """
    def __init__(self, weight):
        super().__init__(weight)
        self.type = "collision"

    def __call__(self, observation_after, action, relative_pose, found_targets, new_voxels, delta_to_target):
        collided = ET.fromstring(observation_after.metadata).find("collision").attrib["status"].lower() == "true"
        return self._weight if collided else 0


class RewardByPickup(RewardBase):
    """Reward if collected, punish if not collected or missed """
    def __init__(self, weight, precision_loss):
        super().__init__(weight)
        self.precision_loss = precision_loss
        assert precision_loss < 0
        self.type = "pickup"

    def __call__(self, observation_after, action, relative_pose, found_targets, new_voxels, delta_to_target):
        # only compute reward and loss if pickup action is selected.
        if action == 3:
            return len(found_targets)* self._weight if len(found_targets) > 0 else self.precision_loss
        return 0

class RewardByObservation(RewardBase):
    """Reward if more target pixels are observed"""
    def __init__(self, weight, saturation_ratio, min_pixel):
        super().__init__(weight)
        self.saturation = saturation_ratio
        self.min_pixel = min_pixel # deal with noise
        self.TARGET = 10
        self.type = "active_perception"


    def __call__(self, observation_after, action, relative_pose, found_targets, new_voxels, delta_to_target):
        # Because the limitation of the simulater, pickup action will not update the fruit observation. To avoid
        # changing the infrastructure, just disable the reward when pickup is selected.
        if action == 3:
            return 0
        _, segmentation, _ = observation_after.images
        seg, _, _ = np.split(segmentation, 3, axis=2)
        seg = np.squeeze(seg)
        # Number of pixels observed as target class from raw observation after the move.
        found_pixels = np.sum(seg == self.TARGET)
        if found_pixels <= self.min_pixel:
            return 0
        ratio = np.clip(found_pixels / seg.size, self.min_pixel/ seg.size, self.saturation)
        # Saturate at selected max ratio.
        return (ratio / self.saturation) * self._weight


class RewardByExploration(RewardBase):
    """ Reward if new voxels are observed """
    def __init__(self, weight, max_discovery):
        super().__init__(weight)
        self.max_discovery = max_discovery
        self.type = "exploration"

    def __call__(self, observation_after, action, relative_pose, found_targets, new_voxels, delta_to_target):
        new_voxels = np.clip(new_voxels, 0, self.max_discovery)
        return (new_voxels / self.max_discovery) * self._weight


def construct_reward(reward_config):
    """Factory method that construct the reward"""
    reward_type = reward_config['type']
    weight = reward_config['weight']
    if reward_type == 'by_forward_movement':
        return RewardByForwardMovement(weight)
    elif reward_type == 'by_distance_traveled':
        return RewardByDistanceTraveled(weight)
    elif reward_type == 'by_collision':
        return RewardByCollision(weight)
    elif reward_type == 'by_decay':
        return RewardByDecay(weight)
    elif reward_type == 'by_pickup':
        precision_loss = reward_config['precision_loss']
        return RewardByPickup(weight, precision_loss)
    elif reward_type == 'by_observation':
        saturation = reward_config['saturation_ratio']
        min_pixel = reward_config['min_pixel']
        return RewardByObservation(weight, saturation, min_pixel)
    elif reward_type == 'by_exploration':
        max_discovery = reward_config['max_discovery']
        return RewardByExploration(weight, max_discovery)
    elif reward_type == 'by_target_pursuit':
        angular_weight = reward_config['angular_weight']
        return RewardByTargetPursuit(weight, angular_weight)
    else:
        raise NotImplementedError("The selected reward term " + reward_type + " is not implemented")