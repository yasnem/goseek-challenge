import open3d as o3d
from perception.tsdf_tools import VoxelGrid
from tests.test_helper import *
from common.visualization import AgentViz
from common.tools import get_config

grid = VoxelGrid(get_config("debug")["voxel_grid"])

agent_viz = AgentViz()
for i in range(4):
    # First get the reprojected point cloud for visualization only!
    cloud_w = grid.get_rgbd_cloud(*get_test_input(i))
    # Integrate local frame.
    grid.integrate(*get_test_input())
    # Visualize the 3D mesh reconstruction.
    mesh = grid.get_mesh()
    # Visualize the tsdf voxel grid.
    tsdf_in_grid = grid.get_tsdf_voxel_grid()
    # Visualize the tsdf voxel grid.
    cloud_from_voxel = grid.get_cloud()
    # View the current state of map with frame input.
    o3d.visualization.draw_geometries([cloud_from_voxel, mesh, tsdf_in_grid])
    # Retrieve voxels from the local frame.
    grid.observe_local(get_pose(i))
    # Extract the tsdf features.
    tsdf = grid.get_tsdf()
    # Visualize the tsdf grid as colored cloud.
    tsdf_color = grid.feature_as_cloud(tsdf)
    # Also visualize the agent's 3D location.
    o3d.visualization.draw_geometries([mesh, tsdf_color, *agent_viz.get_visuals(get_pose(i))])
    # Visualize the segmentation results.
    segmentations = grid.get_dominant_segmentation()
    o3d.visualization.draw_geometries([mesh, grid.feature_as_cloud(segmentations), *agent_viz.get_visuals(get_pose(i))])
    # Visualize the weight results.
    weights = grid.get_weights()
    o3d.visualization.draw_geometries([mesh,  grid.feature_as_cloud(weights), *agent_viz.get_visuals(get_pose(i))])
    # Visualize the frontal frontier results with current pose.
    frontier = grid.get_fovfrontier(get_pose(i))
    target = grid.get_target(get_pose())
    o3d.visualization.draw_geometries([mesh, frontier, *agent_viz.get_visuals(get_pose(i)),
                                       *agent_viz.get_visuals([target[0], target[1], 0])])
    # Visualize the frontier results.
    frontier = grid.get_frontier()
    o3d.visualization.draw_geometries([mesh, frontier, *agent_viz.get_visuals(get_pose(i))])
