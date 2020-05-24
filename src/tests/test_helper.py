import numpy as np
from common.tools import test_data_folder2, convert_pose_stack, extract_segment_class

max_depth = 50
k_meter_to_millimeter = 1000


def get_traj_2D():
    return np.load(test_data_folder2 + "traj.npy")


def get_pose(i=0):
    return get_traj_2D()[i]


def get_traj_transformations():
    """This is the transformation from local to global frame for each step. """
    return convert_pose_stack(get_traj_2D())


def get_rgb_depth_seg(i=0):
    rgb_np = np.load(test_data_folder2 + '/color_{:05d}.npy'.format(i))
    depth_np = np.load(test_data_folder2 + '/depth_{:05d}.npy'.format(i))
    segmentation_np = np.load(test_data_folder2 + '/segment_{:05d}.npy'.format(i))

    return rgb_np, depth_np, extract_segment_class(segmentation_np)


def get_test_input(i=0):
    assert i < 4, "Test data contains 4 frames."
    return (*get_rgb_depth_seg(i), get_pose(i))
