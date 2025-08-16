import numpy as np


def rotational_angle_between_quaternions(quaternion1, quaternion2):
    """
    Get rotational angle between two quaternions
    Inputs:
    - quaternion1: rotation in [w, x, y, z] format
    - quaternion2: rotation in [w, x, y, z] format
    Output:
    - rotational_angle_degrees: rotational angle [deg]
    """
    dot_product = np.dot(quaternion1, quaternion2)
    dot_product = np.clip(abs(dot_product), -1.0, 1.0)
    rotational_angle = 2 * np.arccos(dot_product)
    rotational_angle_degrees = np.degrees(rotational_angle)
    return rotational_angle_degrees

