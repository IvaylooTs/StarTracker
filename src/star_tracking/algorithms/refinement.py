import numpy as np
from ..algorithms.quest import compute_attitude_quaternion
from ..core.coordinates import inverse_rotate_vectors

EPSILON = 1e-3


def calculate_error_radians(image_vectors, catalog_vectors_camera_frame):
    """
    Calculate angular error (in radians) between observed image star vectors
    and reprojected catalog vectors.
    Inputs:
    - image_vectors: Nx3 array of unit vectors from the image (camera frame)
    - catalog_vectors_camera_frame: [N x 3] matrix of catalog vectors rotated
      into the camera frame using the current quaternion
    Output:
    - angular_errors: list of angular differences in radians
    """
    angular_errors = []
    for u_img, u_cat in zip(image_vectors, catalog_vectors_camera_frame):
        dot = np.clip(np.dot(u_img, u_cat), -1.0, 1.0)
        theta = np.arccos(dot)
        angular_errors.append(theta)
    return angular_errors


def calculate_weights_radians(angular_errors):
    """
    Calculate QUEST weights based on angular errors between observed star vectors
    and catalog vectors rotated into camera frame.
    Inputs:
    - angular_errors: list of angular differences in radians between image star vectors
    and catalog vectors in camera frame
    Outputs:
    - weights: array of floats, higher weight for better alignment
    """
    weights = []
    for theta in angular_errors:
        if theta is None or np.isnan(theta):
            weights.append(0.0)
        else:
            weights.append(1.0 / (theta**2 + EPSILON))

    return np.array(weights)


def refine_quaternion(quaternion, catalog_vector_matrix, image_vector_matrix):
    """
    Refine quaternion by running QUEST again with calculated weights
    Inputs:
    - quaternion: rotation in [w, x, y, z] format
    - catalog_vector_matrix: [N x 3] matrix of unit vectors from our catalog
    - image_star_coords: coordinates of the stars in our image
    - image_vector_matrix: [N x 3] matrix of unit vectors from the stars in our image
    Outputs:
    - refined_quaternion: corrected rotation in [w, x, y, z] format
    """

    image_plane_catalog_vectors = inverse_rotate_vectors(
        quaternion, catalog_vector_matrix
    )
    error_rates = calculate_error_radians(
        image_vector_matrix, image_plane_catalog_vectors
    )
    weights = calculate_weights_radians(error_rates)
    refined_quaternion = compute_attitude_quaternion(
        image_vector_matrix, catalog_vector_matrix, weights
    )
    refined_quaternion /= np.linalg.norm(refined_quaternion)
    return refined_quaternion

