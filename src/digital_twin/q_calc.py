import numpy as np

# Given quaternions
q1 = np.array([0.05913049, -0.06343448, 0.71884584, -0.6897393])
q2 = np.array([-0.91246252, -0.39980917, -0.03613716, -0.07911315])

# Normalize them
q1_norm = q1 / np.linalg.norm(q1)
q2_norm = q2 / np.linalg.norm(q2)

# Dot product
dot_product = np.dot(q1_norm, q2_norm)

# Clamp to avoid numerical issues
dot_product = np.clip(dot_product, -1.0, 1.0)

# Angle distance formula for quaternions
angle_rad = 2 * np.arccos(abs(dot_product))
angle_deg = np.degrees(angle_rad)

print(angle_rad, angle_deg)