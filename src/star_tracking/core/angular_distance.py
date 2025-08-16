import numpy as np
from itertools import combinations
from .coordinates import star_coords_to_unit_vector


def angular_dist_helper(unit_vectors):
    """
    Angular distance between each unique pair of unit vectors
    Inputs:
    - unit_vectors: np array containing [x, y, z] coordinates of each unit vector
    Outputs:
    - angular_dists: dict where {(star_index1 [-], star_index2 [-]): angular_distance [deg]}
    """

    # Multiply unit vector matrix by it's transpose to find the dot products
    dot_products = unit_vectors @ unit_vectors.T
    # clip values to an interval of [-1.0, 1.0] (arccos allowed values) to account for float rounding error
    dot_products = np.clip(dot_products, -1.0, 1.0)

    angular_dist_matrix = np.degrees(np.arccos(dot_products))
    # Create a dict with all unique combination from the angular distance matrix
    angular_dists = {
        (i, j): angular_dist_matrix[i, j]
        for i, j in combinations(range(len(unit_vectors)), 2)
    }
    return angular_dists


def get_angular_distances(star_coords, center_coords, f_x, f_y):
    unit_vectors = star_coords_to_unit_vector(star_coords, center_coords, f_x, f_y)
    return angular_dist_helper(unit_vectors)
