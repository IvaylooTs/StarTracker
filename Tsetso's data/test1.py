import numpy as np
from scipy.spatial.transform import Rotation as R

# -------------------------------
# Step 1: Load a small star catalog
# -------------------------------

# Example catalog: [Star ID, RA (deg), Dec (deg)]
catalog = np.array([
    [0, 10.0, 20.0],
    [1, 50.0, 30.0],
    [2, 80.0, -10.0],
    [3, 130.0, 40.0],
    [4, 200.0, -20.0]
])

def radec_to_unitvec(ra_deg, dec_deg):
    ra = np.deg2rad(ra_deg)
    dec = np.deg2rad(dec_deg)
    x = np.cos(ra) * np.cos(dec)
    y = np.sin(ra) * np.cos(dec)
    z = np.sin(dec)
    return np.stack((x, y, z), axis=-1)

catalog_vectors = radec_to_unitvec(catalog[:, 1], catalog[:, 2])

# -------------------------------
# Step 2: Simulate observed stars (rotate catalog)
# -------------------------------

# Simulate the spacecraft being rotated 30° around Y-axis
true_rotation = R.from_euler('y', 30, degrees=True)
observed_vectors = true_rotation.apply(catalog_vectors)

# Add small noise
observed_vectors += np.random.normal(0, 0.001, observed_vectors.shape)
observed_vectors = observed_vectors / np.linalg.norm(observed_vectors, axis=1)[:, np.newaxis]

# -------------------------------
# Step 3: Star Matching (brute-force)
# -------------------------------

matches = []
for i, v_obs in enumerate(observed_vectors):
    best_j = -1
    best_cos_angle = -1
    for j, v_cat in enumerate(catalog_vectors):
        cos_angle = np.dot(v_obs, v_cat)
        if cos_angle > best_cos_angle:
            best_j = j
            best_cos_angle = cos_angle
    matches.append((i, best_j))

# -------------------------------
# Step 4: Solve Wahba’s Problem with SVD
# -------------------------------

B = np.zeros((3, 3))
for i, j in matches:
    v_obs = observed_vectors[i]
    v_cat = catalog_vectors[j]
    B += np.outer(v_obs, v_cat)

U, S, Vt = np.linalg.svd(B)
R_est = U @ Vt

# Ensure a proper rotation (det(R)=+1)
if np.linalg.det(R_est) < 0:
    Vt[2, :] *= -1
    R_est = U @ Vt

rotation_est = R.from_matrix(R_est)
euler_est = rotation_est.as_euler('xyz', degrees=True)
quat_est = rotation_est.as_quat()

# -------------------------------
# Output
# -------------------------------

print("Estimated Euler angles (degrees):", euler_est)
print("Estimated Quaternion [x, y, z, w]:", quat_est)
print("True Euler angles (should be ~[0, 30, 0]):", R.from_matrix(true_rotation.as_matrix()).as_euler('xyz', degrees=True))
