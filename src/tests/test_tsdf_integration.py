import open3d as o3d
import numpy as np
from common.tools import test_data_folder2, convert_pose_stack, extract_segment_class
from perception.camera import GoseekCamera

robot_poses_2d = np.load(test_data_folder2 + "traj.npy")
robot_poses = convert_pose_stack(robot_poses_2d)
volume = o3d.integration.UniformTSDFVolume(
    length=8.0,
    resolution=512,
    sdf_trunc=0.04,
    color_type=o3d.integration.TSDFVolumeColorType.RGB8,
    origin=np.array([-4.0, -4.0, -4.0]))
volume2 = o3d.integration.ScalableTSDFVolume(
    voxel_length=4.0 / 512.0,
    sdf_trunc=0.04,
    color_type=o3d.integration.TSDFVolumeColorType.RGB8)

max_depth = 50
k_meter_to_millimeter = 1000

cam = GoseekCamera()

for i in range(4):
    print("Integrate {:d}-th image into the volume.".format(i))
    rgb_np = np.load(test_data_folder2 + '/color_{:05d}.npy'.format(i))
    # Convert to uint RGB
    color = o3d.geometry.Image(rgb_np)
    # Open3D is mm based.
    depth_np = np.load(test_data_folder2 + '/depth_{:05d}.npy'.format(i))
    depth = o3d.geometry.Image((k_meter_to_millimeter * max_depth * depth_np).astype('float32'))

    # Convert from color image to class labels.
    segmentation_np = np.load(test_data_folder2 + '/segment_{:05d}.npy'.format(i))
    # Convert to single channel np, force-copy for consecutive buffer, and convert to open3d image type.
    segmentation = o3d.geometry.Image(extract_segment_class(segmentation_np))
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_trunc=10.0, convert_rgb_to_intensity=False)
    rgbdseg = o3d.geometry.RGBDSegImage.create_from_color_depth_segmentation(color, depth,
                                                                             segmentation, depth_trunc=10.0,
                                                                             convert_rgb_to_intensity=False)

    pcd_W = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd, intrinsic=cam.intrinsic, extrinsic=np.linalg.inv(np.matmul(cam.extrinsic, robot_poses[i])))
    o3d.io.write_point_cloud("data2/colored_cloud_{:05d}.ply".format(i), pcd_W)

    volume.integrate(rgbd, cam.intrinsic, np.linalg.inv(np.matmul(cam.extrinsic, robot_poses[i])))
    volume2.integrateRGBDSeg(rgbdseg, cam.intrinsic, np.linalg.inv(np.matmul(cam.extrinsic, robot_poses[i])))


print("Extract a triangle mesh from the volume1 and visualize it.")
mesh = volume.extract_triangle_mesh()
mesh.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh])
print("Extract a triangle mesh from the volume2 and visualize it.")
mesh2 = volume2.extract_triangle_mesh()
mesh2.compute_vertex_normals()
o3d.visualization.draw_geometries([mesh2])
