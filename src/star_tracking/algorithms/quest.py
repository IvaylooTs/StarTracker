import numpy as np

def build_attitude_profile_matrix(image_vectors, catalog_vectors, weights=None):
    """
    Function that approximates the rotational relationship between the two sets of vectors (image and catalog)
    in a matrix using the weighted sum of their outer products (You can represent any
    full-rank matrix as a sum of outer products of vectors)
    Inputs:
    - image_vectors: np array of the unit vectors from the image.
    - catalog_vectors: np array of the unit vectors from the catalog
    - weights: array of floats. Represent how much we can trust each pair of image to catalog vectors
    Outputs:
    - attitude_profile_matrix: 3x3 matrix that is the sum of the outer product of each pair of vectors,
    multiplied by the pair's weight (attitude profile matrix)
    """

    assert (
        image_vectors.shape == catalog_vectors.shape
    ), "Shape mismatch between image and catalog vectors"

    number_of_rows = image_vectors.shape[0]
    if weights is None:
        weights = np.ones(number_of_rows)

    attitude_profile_matrix = np.zeros((3, 3))
    for i in range(0, number_of_rows):
        attitude_profile_matrix += weights[i] * np.outer(
            image_vectors[i], catalog_vectors[i]
        )

    return attitude_profile_matrix


def build_quaternion_attitude_profile_matrix(attitude_profile_matrix):
    """
    Function that transforms Wahba's problem into quaternion form
    I don't fully understand the math behind why this matrix is constructed exactly in this way,
    but I got the end construction from the end of this lecture:
    https://www.youtube.com/watch?v=BL4KLCEBUUs
    Inputs:
    - attitude_profile_matrix: 3x3 matrix
    Outputs:
    - quaternion_attitude_profile_matrix: 4x4 matrix
    """
    symmetric_part = attitude_profile_matrix + attitude_profile_matrix.T
    trace = np.trace(attitude_profile_matrix)
    antisymmetric_part = np.array(
        [
            attitude_profile_matrix[1, 2] - attitude_profile_matrix[2, 1],
            attitude_profile_matrix[2, 0] - attitude_profile_matrix[0, 2],
            attitude_profile_matrix[0, 1] - attitude_profile_matrix[1, 0],
        ]
    )

    quaternion_attitude_profile_matrix = np.zeros((4, 4))
    quaternion_attitude_profile_matrix[0, 0] = trace
    quaternion_attitude_profile_matrix[0, 1:] = antisymmetric_part
    quaternion_attitude_profile_matrix[1:, 0] = antisymmetric_part
    quaternion_attitude_profile_matrix[1:, 1:] = symmetric_part - trace * np.eye(3)
    return quaternion_attitude_profile_matrix


def find_optimal_quaternion(quaternion_attitude_profile_matrix):
    """
    Wahba's problem can be rewritten as maximizing this form: q^TKq
    where K is the quaternion_attitude_profile_matrix
    We solve this by finding the eigenvector with the maximum
    eigenvalue which then will correspond to the quaternion
    that best aligns with our rotation
    Inputs:
    - quaternion_attitude_profile_matrix: 4x4 matrix
    Outputs:
    - optimal_quaternion: quaternion that corresponds to our rotation [qw, qx, qy, qz]
    """
    eigenvalues, eigenvectors = np.linalg.eigh(quaternion_attitude_profile_matrix)
    max_index = np.argmax(eigenvalues)
    optimal_quaternion = eigenvectors[:, max_index]
    optimal_quaternion /= np.linalg.norm(optimal_quaternion)
    return optimal_quaternion


def compute_attitude_quaternion(image_vectors, catalog_vectors, weights=None):
    """
    Get orientation in quaternion form
    Inputs:
    - image_vectors: np array of the unit vectors from the image.
    - catalog_vectors: np array of the unit vectors from the catalog
    - weights: how much we trust each vector pair
    Outputs:
    - quaternion: rotation in [w, x, y, z] format
    """
    attitude_profile = build_attitude_profile_matrix(
        image_vectors, catalog_vectors, weights
    )
    quaternion_attitude_profile_matrix = build_quaternion_attitude_profile_matrix(
        attitude_profile
    )
    quaternion = find_optimal_quaternion(quaternion_attitude_profile_matrix)

    return quaternion
