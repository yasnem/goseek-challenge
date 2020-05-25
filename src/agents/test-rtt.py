from src.agents.rrt import RRT
import random
import yaml
from src.perception.tsdf_tools import VoxelGrid
from anytree import PreOrderIter

config = yaml.load(open("/home/jd/competition/shallow-agent/goseek-challenge/baselines/config/nbv-agent.yaml"))
grid = VoxelGrid(config["voxel_grid"])

rrt = RRT(5, 3, grid, [0, 0, 0], 0.99, True)

# For testing purpose, assign some random rewards on the node. 
for n in PreOrderIter(rrt.tree):
    n.reward = random.randint(1, 10)

found, best_reward, next_action = rrt.visit()
print("Found best ", found, " path ", next_action)
