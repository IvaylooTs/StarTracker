# math.py
from scipy.spatial.transform import Rotation as R

import numpy as np
normalization_threshold = 1e-2


def inverse_quaternion(w, x, y, z):
    norm_sq = w**2 + x**2 + y**2 + z**2
    if norm_sq == 0:
        return 1, 0, 0, 0  # Avoid division by zero
    return (
        w / norm_sq,
        -x / norm_sq,
        -y / norm_sq,
        -z / norm_sq
    )

def is_normalized(w, x, y, z):
    norm = w**2 + x**2 + y**2 + z**2
    return abs(norm - 1.0) < normalization_threshold  # Check if norm is close to 1.0
def multiply_quaternions(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return (w, x, y, z)

def round_quaternion(q, decimals=6):
    return tuple(round(c, decimals) for c in q)



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
    # return quaternion_angle_ignore_roll(quaternion1, quaternion2)

def quaternion_angle_ignore_roll(q1, q2, degrees=True):
    """
    Compute the angle between two quaternions, ignoring roll (rotation about the forward axis).
    
    Parameters:
        q1, q2 : array-like, shape (4,)
            Quaternions in [w, x, y, z] format.
        degrees : bool
            If True, returns the angle in degrees. Otherwise in radians.
    
    Returns:
        angle : float
            The angular difference between q1 and q2 ignoring roll.
    """
    # Convert to [x, y, z, w] format for scipy
    q1_xyzw = [q1[1], q1[2], q1[3], q1[0]]
    q2_xyzw = [q2[1], q2[2], q2[3], q2[0]]

    # Convert quaternions to scipy Rotations
    r1 = R.from_quat(q1_xyzw)
    r2 = R.from_quat(q2_xyzw)

    # Extract yaw, pitch, roll (in radians)
    yaw1, pitch1, _ = r1.as_euler('YXZ', degrees=False)  
    yaw2, pitch2, _ = r2.as_euler('YXZ', degrees=False)

    # Rebuild quaternions without roll
    r1_no_roll = R.from_euler('YX', [yaw1, pitch1], degrees=False)
    r2_no_roll = R.from_euler('YX', [yaw2, pitch2], degrees=False)

    # Relative rotation (difference)
    r_diff = r1_no_roll.inv() * r2_no_roll

    # Get angle of difference
    angle = r_diff.magnitude()
    if degrees:
        angle = np.degrees(angle)

    return angle


# Example usage
q1 = [1, 0, 0, 0]  # Identity in [w, x, y, z]
q2 = R.from_euler('zyx', [45, 20, 30], degrees=True).as_quat()
q2_wxyz = [q2[3], q2[0], q2[1], q2[2]]  # convert to [w,x,y,z]

print("Angle difference (ignoring roll):", quaternion_angle_ignore_roll(q1, q2_wxyz), "degrees")
