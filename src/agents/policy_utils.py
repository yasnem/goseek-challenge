import numpy as np
from src.perception.camera import GoseekCamera


class PickupAgent():
    def __init__(self, pickup_config, camera_config):
        self.min_observations = pickup_config['min_observations']
        self.threshold = pickup_config['distance_threshold']
        self.min_dist = pickup_config['min_dist']
        self.cam = GoseekCamera(camera_config)
        self.inv_z = np.zeros((self.cam.height, self.cam.width))
        for u in range(self.cam.width):
            for v in range(self.cam.height):
                self.inv_z[v, u] = 1.0 / self.cam.unit_vectors[v, u, 2]
        self.target = 10
        """
        Somehow the simulator fucked up it's field of view in horizontal direction, and the `ground truth` is thus
        underestimated. This mean we manually shrink the horizontal field of view by limiting observation in left
        and right edges. Shrink by 15% both sides -> 24 cols both sides.
        """
        self.image_mask = np.ones((self.cam.height, self.cam.width))
        self.image_mask[:, :25] = 0
        self.image_mask[:, -25:] = 0

    def decide_pickup(self, segmentation, depth):
        """
        Based on the segmentation information and depth information, using the camera model, determine whether picking up
        will be successful.
        :param segmentation: segmentation information from the simulator.
        :param depth: depth information from the simulator.
        :return: bool for whether or not should use pickup
        """
        if np.any(segmentation == self.target):
            depth = self.cam.max_depth_m * depth
            distance = np.multiply(self.inv_z, depth)
            return np.sum(np.logical_and(np.logical_and(segmentation == self.target, self.image_mask),
                          np.logical_and(self.min_dist <= distance, distance <= self.threshold))) >= self.min_observations
        else:
            return False
