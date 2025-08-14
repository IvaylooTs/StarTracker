import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example quaternion list: (w, x, y, z)
quaternions = [
    (1.0000, 0.0, 0.0, 0.0000),    # 0°
    (0.9659, 0.0, 0.0, 0.2588),    # 30°
    (0.8660, 0.0, 0.0, 0.5000),    # 60°
    (0.7071, 0.0, 0.0, 0.7071),    # 90°
]

def quat_to_matrix(q):
    """Convert quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ])

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Original X-axis vector
x_vec = np.array([1, 0, 0])

# Draw each quaternion's rotated X-axis as an arrow
for i, q in enumerate(quaternions):
    R = quat_to_matrix(q)
    rotated_vec = R @ x_vec
    ax.quiver(0, 0, 0, rotated_vec[0], rotated_vec[1], rotated_vec[2],
              length=1.0, normalize=True, color=plt.cm.jet(i / len(quaternions)))
    ax.text(rotated_vec[0], rotated_vec[1], rotated_vec[2], f"Q{i+1}")

# Set axis properties
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Quaternion Rotation of X-axis Around Z')

plt.show()
