from .rrt import RRT 

from src.perception.tsdf_tools import VoxelGrid

rrt = RRT(5, 2, VoxelGrid(), [0, 0, 0], 0.99, True)
