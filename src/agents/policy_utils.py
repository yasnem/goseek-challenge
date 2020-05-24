import numpy as np
import torch
import torch.nn as nn
import gym
from stable_baselines3.common.policies import BaseFeaturesExtractor
from perception.camera import GoseekCamera


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
        segmentation, _, _ = np.split(segmentation, 3, axis=2) # Only first channel.
        segmentation = np.squeeze(segmentation)
        if np.any(segmentation == self.target):
            depth = self.cam.max_depth_m * depth
            distance = np.multiply(self.inv_z, depth)
            return np.sum(np.logical_and(np.logical_and(segmentation == self.target, self.image_mask),
                          np.logical_and(self.min_dist <= distance, distance <= self.threshold))) >= self.min_observations
        else:
            return False


class TSDFCNN(BaseFeaturesExtractor):
    """
    CNN from DQN nature paper: https://arxiv.org/abs/1312.5602
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box,
                 cnn_arch, features_dim: int = 512):
        super(TSDFCNN, self).__init__(observation_space, features_dim + 3)
        assert len(cnn_arch) > 0, "Need at least one conv layer!"
        n_input_channels = observation_space.shape[0]
        cnn_modules = [self._conv_layer_set(n_input_channels, cnn_arch[0])]
        for idx in range(len(cnn_arch) - 1):
            cnn_modules.append(self._conv_layer_set(cnn_arch[idx], cnn_arch[idx+1]))
        self.cnn = nn.Sequential(*cnn_modules)

        # Compute shape by doing one forward pass
        with torch.no_grad():
            obs_sample = torch.as_tensor(observation_space.sample()[None])
            tsdf, pose = self.decode_obs(obs_sample)
            n_flatten = self.cnn(torch.as_tensor(tsdf))
            flat = n_flatten.view(n_flatten.size()[0], -1)

        # self.linear2 = nn.Sequential(nn.Linear(flat.size()[1], features_dim),
        #                              nn.LeakyReLU(),
        #                              nn.BatchNorm1d(128),
        #                              nn.Dropout(p=0.15),
        #                              nn.Flatten())
        self.linear = nn.Sequential(nn.Linear(flat.size()[1], features_dim),
                                    nn.ReLU())

    def decode_obs(self, obs):
        tsdf = obs[:, :, :-1, :, :]
        pose = obs[:, 0, -1, :3, 0]
        return tsdf, pose

    def _conv_layer_set(self, in_c, out_c):
        conv_layer = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 3, 3), padding=(0, 1, 0)),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        tsdf, pose = self.decode_obs(observations)
        x = self.cnn(tsdf)
        # flatten cnn output
        x = x.view(x.size()[0], -1)
        out = torch.cat((self.linear(x), pose), 1)
        return out
