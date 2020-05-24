from scipy.spatial.transform import Rotation as R
import numpy as np
import os
import math


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
