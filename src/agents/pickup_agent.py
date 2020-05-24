import numpy as np
from agents.agent import SACAgent
import open3d as o3d


class PickUpAgent(SACAgent):
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
        rgb, segmentation, depth = self.model.env.observe().images
        # create aliases
        camera = self.model.env.grid.camera
        grid = self.model.env.grid
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth,
                                                                  depth_trunc=2.0,
                                                                  convert_rgb_to_intensity=False)

        cloud = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd,
                                                               intrinsic=camera.intrinsic,
                                                               project_valid_depth_only=False)

        np.norm(cloud.points[v * image.width + u]) if cloud.points[
            v * image.witdh + u]
        observation = np.repeat(observation[np.newaxis], 1, 0)
        actions, state = self.model.predict(
            observation, state=self.state, deterministic=False
        )
        self.state = state  # update model state
        return int(actions[0])
