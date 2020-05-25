from anytree import NodeMixin, RenderTree, LevelOrderIter
from dataclasses import dataclass
import copy
import random
import math
from src.perception.tsdf_tools import VoxelGrid, viz_two_grid
# This is 8 degrees
angular_step = 8 / 180 * math.pi
forward_step = 0.5
angular_choices = int((360 - 8) / 8)


@dataclass
class Node:
    # relative action to previous step. 0 is forward, 1 - 44 are turns, -1 is root
    action: int
    # current voxel grid.
    grid: VoxelGrid
    # absolute pose is a 3d list, x, z, yaw.  
    pose: list
    # level.
    level: int
    # new observation compared to previous step.
    reward: int


class TreeNode(Node, NodeMixin):
    def __init__(self, name, action, grid, pose, level, reward, parent=None, children=None):
        super(TreeNode, self).__init__(action=action, grid=grid, pose=pose, level=level, reward=reward)
        self.name = name
        self.parent = parent
        if children:
            self.children = children

    # Punish rotation more than forward motion.
    def steps_needed(self):
        return 1 if self.action == 0 else 2


def after_action(pose, action):
    new_pose = pose.copy()
    if action == 0:
        # Move forward.
        new_pose[0] += forward_step * math.sin(pose[2])
        new_pose[1] += forward_step * math.cos(pose[2])
    else:
        new_pose[2] += action * angular_step
        if new_pose[2] > math.pi:
            new_pose[2] -= 2 * math.pi
    return new_pose


# Pretty print for computed path
def prettify_path(path):
    sequence = [n.action for n in reversed(path)]
    def getActionMove(s):
        if s==0:
            return "go straight --> "
        elif s==-1:
            return "root --> "
        elif s<=22:
            return "turn right " + str(s) + " times --> "
        else:
            return "turn left " + str(45-s) + " times --> "

    return "".join([getActionMove(s) for s in sequence])

class RRT:
    def __init__(self, branch, depth, initial_grid, pose, decay, verbose=False):
        self.decay = decay
        self.tree = TreeNode(name='root', action=-1, grid=initial_grid, pose=pose, level=0, reward=0)
        # keep track of current tail for expanding. 
        self.leaves = [self.tree]
        # branch size. 
        self.branch = branch
        # keep track of current level. 
        self.level = 0
        # Verbosity 
        self.verbose = verbose
        # Maximum trial if there is not enough options.
        self.max_trial = self.branch * 5
        # Safety distance.
        self.safety = 0.2
        # Grow to desired size.
        self.grow(level=depth)

    def visualize(self):
        for pre, fill, node in RenderTree(self.tree):
            treestr = u"%s%s" % (pre, node.name)
            print(treestr.ljust(8), node.action, node.reward, node.pose)

    def grow(self, level=1):
        for ith_level in range(self.level + 1, self.level + 1 + level):
            new_leaves = []
            for node in self.leaves:
                # Grow the given leaf.
                # Attempt to move forward, check that the current is free.
                if node.grid.is_free_ahead(node.pose, self.safety, unobserved_as_free=True):
                    cur_name = 'L' + str(ith_level) + "#" + str(len(new_leaves)) + "Step"
                    leaf_node = self._grow_from(node, action=0, name=cur_name, level=ith_level)
                    new_leaves.append(leaf_node)

                # A root leaf or a translational leaf has to sample it's branches. 
                if node.action <= 0:
                    # store everything sampled so far and do not sample again.
                    sampled = []
                    trial = 0
                    while len(sampled) < self.branch and trial <= self.max_trial:
                        trial += 1
                        possible_action = random.randint(1, angular_choices)
                        new_pose = after_action(node.pose, possible_action)
                        # Only rotate to that direction if later it will be free.
                        if possible_action not in sampled and \
                                node.grid.is_free_ahead(new_pose, self.safety, unobserved_as_free=True):
                            sampled.append(possible_action)
                            cur_name = 'L' + str(ith_level) + "#" + str(len(new_leaves)) + "Rot"
                            leaf_node = self._grow_from(node, action=possible_action, name=cur_name, level=ith_level)
                            new_leaves.append(leaf_node)
            self.leaves = new_leaves.copy()
        self.level += level

    # Make sure to copy the grid from previous node.
    def _grow_from(self, node, action, name, level):
        new_pose = after_action(node.pose, action)
        cur_observed = node.grid.get_observed_voxels()
        nxt_grid = copy.deepcopy(node.grid)
        nxt_grid.integrate_rgbd(nxt_grid.render_from(new_pose), new_pose)
        cur_reward = nxt_grid.get_observed_voxels() - cur_observed

        return TreeNode(name=name, action=action, grid=nxt_grid, pose=new_pose,
                        level=level, reward=cur_reward, parent=node)

    # Traverse whole treed by recursive calls, compute best reward and store path
    def visit(self):
        if self.verbose:
            self.visualize()
        # All reward zero, fails.
        if not any([node.reward for node in LevelOrderIter(self.tree)]):
            return False, 0, 0

        # Go through tree and compute nbv.
        def get_best_path(node, decay):
            if not len(node.children):
                return [node], node.reward * (decay ** node.steps_needed())
            rewards = [get_best_path(n, decay) for n in node.children]
            sub_path, best_reward = max(rewards, key=lambda x: x[1])
            sub_path.append(node)
            return sub_path, (best_reward + node.reward) * (decay ** node.steps_needed())

        path, reward = get_best_path(self.tree, self.decay)
        print(self.level)
        if self.verbose:
            print("The best path: ", prettify_path(path))
        # Get the node just before the root.
        return True, reward, path[-2].action
