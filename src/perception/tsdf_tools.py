import numpy as np
import open3d as od
from common.tools import pose_to_transformation, map_to_RGB
from perception.camera import GoseekCamera
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
        self.channels = config['channels']
        self._width_m = config_local.get('width_m', 0.6)  # region of interest
        self._depth_m = config_local.get('depth_m', 0.6)  # region of interest
        # The coordinate origin is at robot top height, which is 0.5 meters above floor, y pointing downwards.
        self._height_upper_m = config_local.get('height_upper_m', 0.5)  # region of interest
        self._height_lower_m = config_local.get('height_lower_m', -0.1)  # region of interest
        self._height_robot_m = 0.0  # relative height of robot
        self._height_m = self._height_upper_m - self._height_lower_m
        self._max_weight = config_local.get('max_weight', 10.0) # Maximum weight inside the grid.
        # Homogenous coordinate for origin in local frame.
        self._origin_robot = np.array([-self._width_m/2, self._height_lower_m, 0, 1])
        self.grid_size = (int(np.ceil(self._width_m / self._voxel_length_m)),
                          int(np.ceil(self._height_m / self._voxel_length_m)),
                          int(np.ceil(self._depth_m / self._voxel_length_m)))
        self.compute_target = config['compute_target']
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
                origin=np.array([-self.full_side_length/2, -self.full_side_length/2, 0]))  # Minimum possible values.
        else:
            self._volume = od.integration.ScalableTSDFVolume(
                voxel_length=self._voxel_length_m,
                sdf_trunc=self._truncation_m,
                color_type=od.integration.TSDFVolumeColorType.RGB8)

        # Configure the fov for frontier computation.
        self._volume.setFOV(self.camera.h_fov, self.camera.fov,
                            self._height_robot_m, self._height_lower_m, self.camera.depth_truncation_m)

        self.tsdf_grid = None

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
                            self._height_robot_m, self._height_lower_m, self.camera.depth_truncation_m)
        self.tsdf_grid = None

    def integrate(self, rgb, depth, segmentation, pose2d):
        """ Integrate a new depth and rgb image to the current TSDF. """
        self._volume.integrateRGBDSeg(self.camera.convertToO3d(rgb, depth, segmentation), self.camera.intrinsic,
                               np.linalg.inv(np.matmul(self.camera.extrinsic, pose_to_transformation(pose2d))))

    def observe_local(self, pose2d):
        """Request local grid to be extracted"""
        origin = np.matmul(pose_to_transformation(pose2d), self._origin_robot)
        # Get first 3 elements.
        origin = origin[:3]
        self.tsdf_grid = self._volume.extract_tsdf(self._voxel_length_m, self._width_m,
                                                   self._height_m, self._depth_m,
                                                   origin, pose2d[2])

    def get_tsdf(self):
        """ Convert the TSDF to a 3D grid format in area of interest in local frame"""
        # Flatten.
        tsdf = np.asarray([v.tsdf for v in self.tsdf_grid.voxels])
        tsdf = tsdf.reshape(self.tsdf_grid.get_size())
        # TODO(jd): Should we use tsdf equals 0 or weight eqauls zero here?
        # Points that are outside of the explored area (tsdf = 0 by default)
        # should have a value of -1 (totally unobserved)
        # tsdf[np.where(tsdf == 0)] = -1.
        return tsdf

    def get_observed_voxels(self):
        return self._volume.observed_voxels()

    def get_target_likelihood(self):
        """ Compute likelihood of a voxel belonging to specific class"""
        def class_likelihood(segmentations, target_class):
            assert len(segmentations) > target_class
            return 0 if sum(segmentations) == 0 else segmentations[target_class] / sum(segmentations)
        # Flatten
        likelihood = np.asarray([class_likelihood(v.seg, self.target_class) for v in self.tsdf_grid.voxels])
        likelihood = likelihood.reshape(self.tsdf_grid.get_size())
        return likelihood

    def get_weights(self):
        """ Get the weights of individual voxels. """
        weights = np.asarray([v.weight for v in self.tsdf_grid.voxels])
        weights = weights / self._max_weight
        weights = weights.reshape(self.tsdf_grid.get_size())
        # Normalize to be inside max weight.
        return weights

    def get_target(self, pose2d):
        assert self.compute_target
        centroid_flag = self._volume.get_target([pose2d[0], 0.0, pose2d[1]], pose2d[2])
        relative_position = np.asarray([centroid_flag[0] - pose2d[0], centroid_flag[2] - pose2d[1]])
        # This is the desired orientation of the z axis.
        orientation = math.atan2(relative_position[0], relative_position[1])
        print("Use fov frontier: ", centroid_flag[3], " tie breaking: ", centroid_flag[4], " jps status ", centroid_flag[5])
        # orientation -= math.pi /2
        # if orientation > math.pi:
        #     orientation -= 2* math.pi
        # if orientation < -math.pi:
        #     orientation += 2 * math.pi
        return np.asarray([centroid_flag[0], centroid_flag[2], orientation])


    def get_dominant_segmentation(self):
        """ Compute most likely segmentation of a voxel"""
        def most_likely_class(segmentations):
            return self.unobserved_class if sum(segmentations) == 0 else segmentations.index(max(segmentations))
        # Flatten
        seg = np.asarray([most_likely_class(v.seg) for v in self.tsdf_grid.voxels])
        # Convert to -1 to 1.
        seg = seg / self.unobserved_class
        # Into 3D grid format.
        seg = seg.reshape(self.tsdf_grid.get_size())
        return seg

    def get_tsdf_voxel_grid(self):
        """ Visualize the tsdf values stored inside the voxel grid. """
        return self._volume.extract_voxel_point_cloud()

    def get_cloud(self):
        """ Visualize the curent cloud from the voxel grid. """
        return self._volume.extract_point_cloud()

    ### Various Debug functions for visualization.
    def get_point_cloud(self, visualize=False):
        """ Extract the point cloud and possibly visualize it. """
        pointcloud = self._volume.extract_point_cloud()
        if visualize:
            od.draw_geometries([pointcloud])
        return pointcloud

    def get_mesh(self, visualize=False):
        """ Extract the mesh by triangulation and visualize it. """
        mesh = self._volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        if visualize:
            od.visualization.draw_geometries([mesh])
        return mesh

    def get_frontier(self):
        return self._volume.extract_frontier()

    def get_fovfrontier(self, pose2d):
        fov_frontier = self._volume.extract_fov_frontier(np.asarray([pose2d[0], 0.0, pose2d[1]]), pose2d[2])
        # Make it black.
        fov_frontier.paint_uniform_color([0.9, 0.1, 0.1])
        return fov_frontier

    def feature_as_cloud(self, feature):
        """ Visualize a TSDF (by default the last one generated). """
        min_feature = math.floor(np.amin(feature))
        max_feature = math.ceil(np.amax(feature))
        feature = np.clip(feature, min_feature, max_feature)
        vis = od.geometry.PointCloud()

        for (i, j, k), val in np.ndenumerate(feature):
            val = (val-min_feature) / (max_feature - min_feature) if max_feature != min_feature else 0
            # Adding centroid of given index, this is already in global frame!
            vis.points.append(self.tsdf_grid.get_voxel_center(np.array([i, j, k])))
            # Convert to a color map from 0 to 1.
            vis.colors.append(map_to_RGB(val))
        return vis

    def get_rgbd_cloud(self, rgb, depth, segmentation, pose2d):
        return od.geometry.PointCloud.create_from_rgbd_image(image=self.camera.convertToO3dRGBD(rgb, depth),
                                                             intrinsic=self.camera.intrinsic,
                extrinsic=np.linalg.inv(np.matmul(self.camera.extrinsic, pose_to_transformation(pose2d))))
