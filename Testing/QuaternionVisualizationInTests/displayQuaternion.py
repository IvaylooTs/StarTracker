import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example quaternion list: (w, x, y, z)
quaternions = [
    # (1.0000, 0.0, 0.0, 0.0000),    # 0°
    # (0.9659, 0.0, 0.0, 0.2588),    # 30°
    # (0.8660, 0.0, 0.0, 0.5000),    # 60°
    # (0.7071, 0.0, 0.0, 0.7071),    # 90°
    # (1,0,0,0),
    # (0,1,0,0),
    # (0,0,1,0),
    # (0,0,0,1)
    # (-0.4098392 ,  0.57599369, -0.41042687,  0.57603202),
    # ( 0.41006729, -0.57620801, -0.41029401,  0.57574992),
    # ( 1,0,0,0),
    # (0.63936127, -0.27589934,  0.26554076, -0.66677194),

# (-0.48891368,  0.51058666 ,-0.4894919  , 0.51055103),
# ( 0.64845516, -0.6486699  ,-0.28149915 ,-0.28194234),
# (-0.48921604,  0.51073443 , 0.48933815 ,-0.5102609 ),
# ( 0.28166202,  0.281782   , 0.64882765 ,-0.6482963 ),
# ( 0.43029604,  0.17814058 , 0.33914891 ,-0.81736728),
# ( 6.93714809e-01,  5.03989616e-05,  4.05545395e-04 ,-7.20249677e-01),
# (1,0,0,0),
# ( 0.707,0,-0.707,0),
# ( 0.707,0,0, 0.707)
# ( 7.10947858e-01,  3.79553756e-05,  3.97649653e-04, -7.03244612e-01),
# ( 0.2817049,   0.28194615,  0.64878552, -0.64824847),
# (-0.50207073,  0.49761117, -0.50270326,  0.49759174)


(-0.49967516,  0.5000071  ,-0.50029692,  0.50002063), # 0 0
(-0.65324067,  0.65319716 ,-0.2714037 ,  0.27009282), # 45 0
( 0.65311873, -0.27056981 , 0.27098332, -0.65329621), # 0 45
( 0.85384921, -0.35350319 , 0.14722927, -0.35256284), # 45 45
(0.707,0,0.707,0)

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
