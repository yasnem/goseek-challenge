import open3d as o3d
from scipy.spatial.transform import Rotation as R
from common.tools import pose_to_transformation


class AgentViz():
    def __init__(self):
        self.mesh_cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=0.125,
                                                                  height=0.5)
        self.mesh_cylinder.compute_vertex_normals()
        self.mesh_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
        self.mesh_cylinder.rotate(R.from_euler('x', 90, degrees=True).as_matrix())
        self.mesh_cylinder.translate([0, 0.25, 0])
        # Robot frame in world.
        self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.4, origin=[0, 0, 0])
        # Origin of the world frame.
        self.mesh_frame_W = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.25, origin=[0, 0, 0])

    def get_visuals(self, pose2d):
        T = pose_to_transformation(pose2d)
        agent = o3d.geometry.TriangleMesh(self.mesh_cylinder)
        frame_w = o3d.geometry.TriangleMesh(self.mesh_frame_W)
        frame = o3d.geometry.TriangleMesh(self.mesh_frame)
        return agent.transform(T), frame_w.transform(T), frame

    def get_frame(self, pose2d):
        return o3d.geometry.TriangleMesh(self.mesh_frame).transform(pose_to_transformation(pose2d))

    def get_cylinder(self, pose2d):
        return o3d.geometry.TriangleMesh(self.mesh_cylinder).transform(pose_to_transformation(pose2d))