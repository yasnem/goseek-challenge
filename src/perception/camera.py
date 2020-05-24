import math
import open3d as o3d
import numpy as np

class GoseekCamera():
    def __init__(self, config):
        self.width = config.get('width', 320)
        self.height = config.get('height', 240)
        # Field of view is taken from the simulation.
        self.fov = config.get('fov', 60) / 180 * math.pi # in radians, fov in vertical.
        self.h_fov = config.get('h_fov', 80) / 180 * math.pi # in radians, fov in horizontal.
        # https: // docs.unity3d.com / ScriptReference / Camera - aspect.html
        # https://forum.unity.com/threads/how-to-calculate-horizontal-field-of-view.16114/
        # self.fx = self.width / 2 / math.tan(self.fov_h / 2)
        # The horizontal fov should have been 80 degrees (60 times aspect ratio), however, 0.717 fits better as a magic
        # number... With that, I could reach ~mm precision at the edge of the fov, which I consider good enough.
        half_horizontal_fov = 0.717 * (config.get('fov', 60) / 60)  # linearly scale if fov changes.
        self.fx = self.width / 2 / math.tan(half_horizontal_fov)
        self.fy = self.height / 2 / math.tan(self.fov / 2)
        self.intrinsic = o3d.camera.PinholeCameraIntrinsic(self.width, self.height, self.fx, self.fy, self.width / 2,
                                                           self.height / 2)
        # This converts the camera image into  x-right, y-forward, z-up.
        extrinsics = config.get('extrinsics')
        self.extrinsic = np.array([[1.0, 0, 0, extrinsics.get('x', -0.05)], [0, 1.0, 0.0, extrinsics.get('y', 0)],
                                   [0, 0.0, 1.0, extrinsics.get('z', 0.0)], [0, 0, 0, 1.0]], dtype='float64')
        self.max_depth_m = config.get('max_depth_m', 50.0)
        self.k_meter_to_millimeter = 1000
        self.depth_truncation_m = config.get('depth_truncation_m', 5.0)
        self.ransac_dist_thresh_m = config.get('ransac_dist_thresh_m', 0.01)
        self.convert_rgb_to_intensity = False
        self.TARGETCLASS = 10
        self.WALL_CLS = 2
        self.default_invalid_depth = 0.25
        fx, fy = self.intrinsic.get_focal_length()
        ux, uy = self.intrinsic.get_principal_point()
        Kinv = np.linalg.inv(np.asarray([[fx, 0, ux], [0, fy, uy], [0, 0, 1]]))
        self.unit_vectors = np.zeros((self.height, self.width, 3))
        for u in range(self.width):
            for v in range(self.height):
                unit_vector = np.matmul(Kinv, np.array([u, v, 1]))
                self.unit_vectors[v, u, :] = unit_vector / np.linalg.norm(unit_vector)

    def convertToO3d(self, rgb, depth, segmentation):
        depth = self.fillInvalidDepth(segmentation, depth)
        color = o3d.geometry.Image(rgb)
        depth = o3d.geometry.Image((self.k_meter_to_millimeter * self.max_depth_m * depth).astype('float32'))
        # Convert to single channel np, force-copy for consecutive buffer, and convert to open3d image type.
        segmentation = o3d.geometry.Image(segmentation)
        # For invalid segmentations, extra care must be taken.
        return o3d.geometry.RGBDSegImage.create_from_color_depth_segmentation(color, depth, segmentation,
                depth_trunc=self.depth_truncation_m, convert_rgb_to_intensity=self.convert_rgb_to_intensity)


    def convertToO3dRGBD(self, rgb, depth):
        color = o3d.geometry.Image(rgb)
        depth = o3d.geometry.Image((self.k_meter_to_millimeter * self.max_depth_m * depth).astype('float32'))
        return o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth,
                depth_trunc=self.depth_truncation_m, convert_rgb_to_intensity=self.convert_rgb_to_intensity)


    # Plane fitting and filling invalid depth. Look at immeidate neighbors that are classified as wall.
    # This probably needs to be adapted for kimera env.
    def fillInvalidDepth(self, segmentation, depth):
        mask = depth == 0
        # Everything valid
        if np.sum(mask) == 0:
            return depth
        wall = segmentation==self.WALL_CLS
        mask_down = np.roll(mask, 1, axis=0)
        mask_down[0,:] = 0
        mask_up = np.roll(mask, -1, axis=0)
        mask_up[-1,:] = 0
        mask_right = np.roll(mask, 1, axis=1)
        mask_right[:,0] = 0
        mask_left = np.roll(mask, -1, axis=1)
        mask_left[:,-1] = 0
        # Get points that are neighbors to invalid pixels.
        neighbors = np.logical_or(np.logical_or(mask_up, mask_down), np.logical_or(mask_left, mask_right))
        neighbors = np.logical_and(neighbors, np.logical_not(mask))
        neighbors = np.logical_and(neighbors, wall)
        if np.sum(neighbors) < 3:
            print("Too little valid neighbors to estimate proper depth. you are screwed. Assign 0.25 to all.")
            depth[mask] = self.default_invalid_depth / self.max_depth_m
            return depth
        # Neighbors are the point mask that are direct neighbors to the missing segments.
        depth_neighbors = np.zeros_like(depth)
        depth_neighbors[neighbors] = depth[neighbors]
        # Copy only the point that are valid neighbors.
        cloud = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image((self.k_meter_to_millimeter * self.max_depth_m * depth_neighbors).astype('float32')), self.intrinsic)
        plane_model, inliers = cloud.segment_plane(distance_threshold=self.ransac_dist_thresh_m,
                                                 ransac_n=3,
                                                 num_iterations=250)
        # inlier_cloud = cloud.select_by_index(inliers)
        # inlier_cloud.paint_uniform_color([1.0, 0, 0])
        #
        # outlier_cloud = cloud.select_by_index(inliers, invert=True)
        # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
        if np.sum(inliers) < 3:
            print("Ransac failed. you are screwed. Assign 0.25 to all.")
            depth[mask] = self.default_invalid_depth / self.max_depth_m
            return depth
        [a, b, c, d] = plane_model
        if abs(b)>0.02:
            print("doesnot look like a wall. Assing max distance to all.")
            depth[mask] = self.default_invalid_depth / self.max_depth_m
            return depth
        # Now project the plane to image.
        for u in range(self.width):
            for v in range(self.height):
                if mask[v, u]:
                    dist = -d / np.dot(self.unit_vectors[v, u, :], np.asarray([a, b, c])) / self.k_meter_to_millimeter
                    depth[v, u] = dist * self.unit_vectors[v, u, 2] / self.max_depth_m if dist > 0 else 0
        return depth


class CameraPose:
    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return 'Metadata : ' + ' '.join(map(str, self.metadata)) + '\n' + \
               "Pose : " + "\n" + np.array_str(self.pose)
