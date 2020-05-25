import numpy as np
import open3d as od
from src.common.tools import pose_to_transformation, map_to_RGB
from src.common.visualization import AgentViz
from src.perception.camera import GoseekCamera
import math


class VoxelGrid():
    """ Class for manipulating TSDF Voxel grid objects, using Open3D. """

    def __init__(self, config):
        self.camera = GoseekCamera(config['camera'])
        config_tsdf = config['params']
        self._voxel_length_m = config_tsdf.get('voxel_length_m', 0.1)
        self._truncation_m = config_tsdf.get('truncation_m', 0.8)
        assert self._truncation_m < 1, \
            "Currently truncation distance must be smaller than 1 as -1 is considered unobserved"
        config_local = config['local']
        self._width_m = config_local.get('width_m', 0.6)  # region of interest
        self._depth_m = config_local.get('depth_m', 0.6)  # region of interest
        # The coordinate origin is at robot top height, which is 0.5 meters above floor, y pointing downwards.
        self._height_upper_m = config_local.get('height_upper_m', 0.5)  # region of interest
        self._height_lower_m = config_local.get('height_lower_m', -0.1)  # region of interest
        self._height_robot_m = 0.0  # relative height of robot
        self._height_m = self._height_upper_m - self._height_lower_m
        self._max_weight = config_local.get('max_weight', 10.0) # Maximum weight inside the grid.
        # For pixel without observation from the rendering, this depth is assigned to compute new observation.
        self._rendered_invalid = config['render'].get('rendered_invalid', 10.0) # Maximum weight inside the grid.
        # Homogenous coordinate for origin in local frame.
        self._origin_robot = np.array([-self._width_m/2, self._height_lower_m, 0, 1])
        self.grid_size = (int(np.ceil(self._width_m / self._voxel_length_m)),
                          int(np.ceil(self._height_m / self._voxel_length_m)),
                          int(np.ceil(self._depth_m / self._voxel_length_m)))
        # The class index of interest.
        self.target_class = 10
        self.unobserved_class = 11

        # Underlying data structure.
        self._grid_type = config_tsdf.get("type", "scalable")
        if self._grid_type == "uniform":
            self.full_side_length = config_tsdf.get("full_length_m", 10.0)
            self._volume = od.integration.UniformTSDFVolume(
                length=self.full_side_length,
                resolution=math.ceil(self.full_side_length/self._voxel_length_m),
                sdf_trunc=self._truncation_m,
                color_type=od.integration.TSDFVolumeColorType.RGB8,
                origin=np.array([-self.full_side_length/2, -self.full_side_length/2, -self.full_side_length/2]))  # Minimum possible values.
        else:
            self._volume = od.integration.ScalableTSDFVolume(
                voxel_length=self._voxel_length_m,
                sdf_trunc=self._truncation_m,
                color_type=od.integration.TSDFVolumeColorType.RGB8)

        # Configure the fov for frontier computation.
        self._volume.setFOV(self.camera.h_fov, self.camera.fov,
                            self._height_robot_m, self._height_lower_m, self._rendered_invalid)


    def reset(self):
        if self._grid_type == "uniform":
            self._volume = od.integration.UniformTSDFVolume(
                length=self.full_side_length,
                resolution=math.ceil(self.full_side_length/self._voxel_length_m),
                sdf_trunc=self._truncation_m,
                color_type=od.integration.TSDFVolumeColorType.RGB8,
                origin=np.array([-self.full_side_length/2, -self.full_side_length/2, 0]))  # Minimum possible values.
        else:
            self._volume = od.integration.ScalableTSDFVolume(
                voxel_length=self._voxel_length_m,
                sdf_trunc=self._truncation_m,
                color_type=od.integration.TSDFVolumeColorType.RGB8)

        self._volume.setFOV(self.camera.h_fov, self.camera.fov,
                            self._height_robot_m, self._height_lower_m, self._rendered_invalid)

    def integrate(self, rgb, depth, segmentation, pose2d):
        """ Integrate a new depth and rgb image to the current TSDF. """
        """ 0 values are considered freespace! """
        self._volume.integrateRGBDSeg(self.camera.convertToO3d(rgb, depth, segmentation), self.camera.intrinsic,
                               np.linalg.inv(np.matmul(self.camera.extrinsic, pose_to_transformation(pose2d))))

    def integrate_rgbd(self, rgbd, pose2d):
        """ Integrate from a rgbd image """
        self._volume.integrate(rgbd, self.camera.intrinsic, np.linalg.inv(np.matmul(
            self.camera.extrinsic, pose_to_transformation(pose2d))))

    def render_from(self, pose2d):
        """ Render a fake rgbd image from current mesh. """
        mesh = self.get_mesh()
        mesh = mesh.transform(np.linalg.inv(pose_to_transformation(pose2d)))
        rendered_rgbd = mesh.render_rgbd_image(self.camera.intrinsic)
        modify = np.asarray(rendered_rgbd.depth)
        modify[np.where(modify == 0)] = self._rendered_invalid
        return rendered_rgbd

    def observe_local_tsdf(self, pose2d):
        """Return local tsdf grid"""
        origin = np.matmul(pose_to_transformation(pose2d), self._origin_robot)
        # Get first 3 elements.
        origin = origin[:3]
        local_grid = self._volume.extract_tsdf(self._voxel_length_m, self._width_m,
                                                   self._height_m, self._depth_m,
                                                   origin, pose2d[2])
        tsdf = np.asarray([v.tsdf for v in local_grid.voxels])
        return tsdf.reshape(local_grid.get_size())

    def is_free_ahead(self, pose, safety_dist, unobserved_as_free=False):
        # Make sure no collision can happen between the current position and 0.5 m ahead
        tsdf = self.observe_local_tsdf(pose)
        assert tsdf.shape[2] <= (0.5/self._voxel_length_m)
        assert safety_dist > 0
        # Mark the several first depth dimension free because the robot is assumed to be not colliding.
        tsdf[:, :, :2] = 1
        if unobserved_as_free:
            tsdf[np.where(tsdf == 0)] = 1
        return np.all(tsdf > safety_dist)

    def visualize(self, pose):
        od.visualization.draw_geometries([self.get_mesh(),
                                          *AgentViz().get_visuals(pose)])

    def get_observed_voxels(self):
        return self._volume.observed_voxels()

    def get_tsdf_voxel_grid(self):
        """ Visualize the tsdf values stored inside the voxel grid. """
        return self._volume.extract_voxel_point_cloud()

    def get_cloud(self):
        """ Visualize the curent cloud from the voxel grid. """
        return self._volume.extract_point_cloud()

    ### Various Debug functions for visualization.
    def get_point_cloud(self):
        """ Extract the point cloud and possibly visualize it. """
        pointcloud = self._volume.extract_point_cloud()
        return pointcloud

    def get_mesh(self):
        """ Extract the mesh by triangulation and visualize it. """
        mesh = self._volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        return mesh

    def get_rgbd_cloud(self, rgb, depth, segmentation, pose2d):
        return od.geometry.PointCloud.create_from_rgbd_image(image=self.camera.convertToO3dRGBD(rgb, depth),
                                                             intrinsic=self.camera.intrinsic,
                extrinsic=np.linalg.inv(np.matmul(self.camera.extrinsic, pose_to_transformation(pose2d))))
