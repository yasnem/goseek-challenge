import open3d as o3d
from perception.tsdf_tools import VoxelGrid
from tests.test_helper import *
from common.tools import get_config

i = 0

grid = VoxelGrid(get_config("navigation")["voxel_grid"])

cloud_w = grid.get_rgbd_cloud(*get_test_input(i))
# Integrate local frame.
grid.integrate(*get_test_input())
# Visualize the 3D mesh reconstruction.
mesh = grid.get_mesh()

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(mesh)
vis.run()  # user changes the view and press "q" to terminate
param = vis.get_view_control().convert_to_pinhole_camera_parameters()
o3d.io.write_pinhole_camera_parameters("viewpoint.json", param)
vis.destroy_window()


def load_view_point(pcd, filename):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    param = o3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()