from scipy.spatial.transform import Rotation as R
import numpy as np
import os
import math
import yaml

test_data_folder2 = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) + '/tests/data2/'
nav_config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) + '/config/nav-config.yaml'
goseek_config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) + '/config/goseek-config.yaml'
debug_config_file = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)) + '/config/debug-config.yaml'




def pose_to_transformation(pose, degrees=False):
    """
    The 2d pose is under the format [x,z,yaw]
    Yaw is thus around the y axis in degrees
    """
    result = np.eye(4)
    result[0, 3] = pose[0]  # Assign x
    result[2, 3] = pose[1]  # Assign z
    r = R.from_euler('y', pose[2], degrees=degrees)
    result[:3, :3] = r.as_matrix()
    return result


def convert_pose_stack(poses, use_degrees=False):
    return np.stack([pose_to_transformation(p, degrees=use_degrees) for p in poses], axis = 0)


def extract_segment_class(segmentation_np):
    # Output from the segmentation has the following RGB channel values: [#SegmentationClass, #Rvalue, #Bvalue]
    # We extract the R channel for the segmentation class.
    seg, _, _ = np.split(segmentation_np, 3, axis=2)
    seg = np.copy(np.squeeze(seg))
    seg[seg > 10] = 2
    return seg


def load_yaml(file_path):
    """Load a YAML file into a Python dict.

    Args:
      file_path (str): The path to the YAML file.

    Returns:
      A dict with the loaded configuration.
    """
    with open(os.path.expanduser(file_path), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def get_config_file(sim_type):
    if sim_type == "navigation":
        return nav_config_file
    elif sim_type == "goseek":
        return goseek_config_file
    elif sim_type == "debug":
        return debug_config_file
    else:
        return ""

def get_config(sim_type):
    """
    Load default YAML file into a Python dict.
    """
    return load_yaml(get_config_file(sim_type))


def get_rewards_config(sim_type):
    return get_config(sim_type)['rewards']


def get_agent_config(sim_type):
    return get_config(sim_type)['agent']


def get_grid_config(sim_type):
    return get_config(sim_type)['voxel_grid']

def map_to_RGB(i):
    """Colormap from scalar to rgb, Blue is 0 and Red is 1, Green is middle."""
    # https: // www.particleincell.com / 2014 / colormap /
    ## TODO(jd): remove this exception once find out why.
    if (i>1 or i<0 or i!= i):
        return np.array([0, 0, 255], dtype="uint8")
    assert 0 <= i <= 1, "mapping only works from 0 to 1 but {%f}"%(i)
    group = (1-i)*4
    group_id = math.floor(group)
    value = math.floor(255*(group - group_id))
    if not group_id:
        return np.array([255, value, 0], dtype="uint8")
    elif group_id == 1:
        return np.array([255-value, 255, 0], dtype="uint8")
    elif group_id == 2:
        return np.array([0, 255, value], dtype="uint8")
    elif group_id == 3:
        return np.array([0, 255-value, 255], dtype="uint8")
    else:
        return np.array([0, 0, 255], dtype="uint8")
#
# def read_trajectory(filename):
#     # Trajectory loader and reader for the default open3d foramt
#     traj = []
#     with open(filename, 'r') as f:
#         metastr = f.readline()
#         while metastr:
#             metadata = list(map(int, metastr.split()))
#             mat = np.zeros(shape=(4, 4))
#             for i in range(4):
#                 matstr = f.readline()
#                 mat[i, :] = np.fromstring(matstr, dtype=float, sep=' \t')
#             traj.append(CameraPose(metadata, mat))
#             metastr = f.readline()
#     return traj
#
#
# def write_trajectory(traj, filename):
#     with open(filename, 'w') as f:
#         for x in traj:
#             p = x.pose.tolist()
#             f.write(' '.join(map(str, x.metadata)) + '\n')
#             f.write('\n'.join(
#                 ' '.join(map('{0:.12f}'.format, p[i])) for i in range(4)))
#             f.write('\n')