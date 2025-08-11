import cv2
import numpy as np
import sqlite3
import math
from itertools import combinations
from itertools import product
from collections import defaultdict
from scipy.spatial.transform import Rotation as rotate
import sys
import os
import time

image_processing_folder_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "image_processing")
)
sys.path.append(image_processing_folder_path)
import image_processing_v6 as ip

IMAGE_HEIGHT = 1964
IMAGE_WIDTH = 3024
ASPECT_RATIO = IMAGE_HEIGHT / IMAGE_WIDTH
FOV_Y = 53
FOV_X = math.degrees(2 * math.atan(math.tan(math.radians(FOV_Y / 2)) * (1 / ASPECT_RATIO)))
CENTER_X = IMAGE_WIDTH / 2
CENTER_Y = IMAGE_HEIGHT / 2
FOCAL_LENGTH_X = (IMAGE_WIDTH / 2) / math.tan(math.radians(FOV_X / 2))
FOCAL_LENGTH_Y = (IMAGE_HEIGHT / 2) / math.tan(math.radians(FOV_Y / 2))
TOLERANCE = 1
IMAGE_FILE = "./test_images/testing63.png"
NUM_STARS = 15
EPSILON = 1e-6
MIN_SUPPORT = 2
MIN_MATCHES = 5


# Unit vector function -> finds star unit vectros based on star pixel coordinates
# using pinhole projection
# We're only finding the direction where the 3D object lies since we don't know the actual depth.
# Inputs:
# - star_coords: array of tuples that holds (x, y) pixel coordinates of stars in the image
# - center_coords: image center (principal point) coordinates in tuple form: (xc, yc)
# - f_x: const. Describes the scaling factor by x
# - f_y: const. Describes the scaling factor by y
# Outputs:
# - unit_vectors: array of np arrays holding (x, y, z) coordinates of each unit vector
def star_coords_to_unit_vector(star_coords, center_coords, f_x, f_y):
    # Unpack center_coords
    cx, cy = center_coords
    # Initialize empty unit vector array
    unit_vectors = []

    # Reverse pinhole projection to get X, Y, Z coords of each star
    for x, y in star_coords:
        # Pick arbitrary depth for each 3D point
        # (we don't know how far away they really are, we just want their direction)
        z_p = 1

        # Convert pixel coordinates into normalized camera coordinates (back-projection step)
        # These correspond to where the pixel projects onto the image plane at depth z_p
        # Inversing pinhole projection formula for x (back-projecting). Formula: x = f_x * (x_p / z_p) + c_x
        x_p = (x - cx) / f_x
        # Inversing pinhole projection formula for y (back-projecting). Formula: y = f_y * (x_p / z_p) + c_y
        y_p = (y - cy) / f_y

        # array holding [X, Y, Z] coordinates of the direction vector
        direction_vector = np.array([x_p, y_p, z_p])
        # Normalize vector to save on calculations later on
        normalized_vector = direction_vector / np.linalg.norm(direction_vector)
        unit_vectors.append(normalized_vector)

    return np.array(unit_vectors)


# Angular distance between each unique pair of unit vectors
# Inputs:
# - unit_vectors: np array containing [x, y, z] coordinates of each unit vector
# Outputs:
# - angular_dists: dict where {(star_index1 [-], star_index2 [-]): angular_distance [deg]}
def angular_dist_helper(unit_vectors):
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


def load_catalog_unit_vectors(db_path):
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT hip_id, x, y, z FROM Stars")

        hip_to_vector = {}
        for hip_id, x, y, z in cursor.fetchall():
            vector = np.array([x, y, z], dtype=float)
            hip_to_vector[hip_id] = vector

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {}

    finally:
        if conn:
            conn.close()

    return hip_to_vector


# Function that loads a dict from the database where {(HIP ID 1 [-], HIP ID 2 [-]): angular_distance [deg]}
def load_catalog_angular_distances(
    db_path="star_distances_sorted.db", table_name="AngularDistances"
):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = f"SELECT hip1, hip2, angular_distance FROM {table_name}"
    cursor.execute(query)
    rows = cursor.fetchall()

    conn.close()

    return {(row[0], row[1]): row[2] for row in rows}


# Function that returns a dict where {(HIP ID 1 [-], HIP ID 2 [-]): angular_distance [deg]}
# angular_distance ∈ [min_ang_dist, max_ang_dist]
# Inputs:
# - cat_ang_dists: dict where {(HIP ID 1 [-], HIP ID 2 [-]): angular_distance [deg]}
# - bounds[2]: array where bounds[0] = min_ang_dist and bounds[1] = max_ang_dist
def filter_catalog_angular_distances(cat_ang_dists, bounds):
    min_ang_dist, max_ang_dist = bounds
    filtered = {
        pair: ang_dist
        for pair, ang_dist in cat_ang_dists.items()
        if min_ang_dist <= ang_dist <= max_ang_dist
    }
    return filtered


# Function that returns an angular distance interval based on initial angular distance and a tolerance
# Outputs:
# - bounds: tuple where bounds[0] = min_ang_dist and bounds[1] = max_ang_dist
def get_bounds(ang_dist, tolerance):
    return (ang_dist - tolerance, ang_dist + tolerance)


# Function that loads candidate catalog HIP IDs for each image star
# Inputs:
# - num_stars: number of stars from the image
# - angular_distances: dict where {(image star index 1, image star index 2): image angular distance}
# - all_cat_ang_dists: dict where {(HIP ID 1, HIP ID 2): catalog angular distance}
# - tolerance: angular distance tolerance
# Outputs:
# - hypotheses_dict: {image star index: [HIP ID 1, ... HIP ID N]}
def load_hypotheses(angular_distances, all_cat_ang_dists, tolerance):
    matches_counter = defaultdict(int)

    for (s1, s2), ang_dist in angular_distances.items():
        if ang_dist is None:
            continue

        bounds = get_bounds(ang_dist, tolerance)
        cur_cat_dict = filter_catalog_angular_distances(all_cat_ang_dists, bounds)

        for (hip1, hip2), cat_ang_dist in cur_cat_dict.items():
            matches_counter[(s1, hip1)] += 1
            matches_counter[(s1, hip2)] += 1
            matches_counter[(s2, hip1)] += 1
            matches_counter[(s2, hip2)] += 1

    hypotheses_dict = defaultdict(set)
    for (star_idx, hip_id), count in matches_counter.items():
        if count >= MIN_SUPPORT:
            hypotheses_dict[star_idx].add(hip_id)

    return hypotheses_dict


# DFS to produce every possible mapping of image star to catalog star ID (even if not full mappings)
# Inputs:
# - assignment: a dict mapping stars in the image to candidate catalog stars (HIP IDs)
# - image_stars: list of stars detected in the image
# - candidate_hips: dict mapping each image star to a list of candidate catalog HIP IDs
# - image_angular_distances: dict where {(image star index 1, image star index 2): image angular distance}
# - catalog_angular_distances: dict where {(HIP ID 1, HIP ID 2): catalog angular distance}
# - tolerance: angular distance tolerance
# - solutions: a list to collect valid assignments (solutions) - initially empty
def DFS(
    assignment,
    image_stars,
    candidate_hips,
    image_angular_distances,
    catalog_angular_distances,
    tolerance,
    solutions,
):

    # Does current assignment qualify as a solution
    if len(assignment) >= MIN_MATCHES:
        solutions.append(dict(assignment))

    # End of recursion condition if we've matched all stars
    if len(assignment) == len(image_stars):
        return
    

    # Selection of next star for assignment
    unassigned = [s for s in image_stars if s not in assignment]
    current_star = min(unassigned, key=lambda s: len(candidate_hips.get(s, [])))
    

    # Main loop for trying candidates for current star
    for hip_candidate in candidate_hips.get(current_star, []):

        # We don't want to assign the same HIP ID twice
        if hip_candidate in assignment.values():
            continue

        assignment[current_star] = hip_candidate

        # Check if the assignment is consistent and recurse to assign next star if so
        if is_consistent(
            assignment,
            image_angular_distances,
            catalog_angular_distances,
            tolerance,
            current_star,
        ):
            DFS(
                assignment,
                image_stars,
                candidate_hips,
                image_angular_distances,
                catalog_angular_distances,
                tolerance,
                solutions,
            )

        del assignment[current_star]


# Bool function that checks if the current assignment is consistent with the angular distances from our image within a certain tolerance
# Inputs:
# - assignment: current assignment
# - image_angular_distances: dict where {(image star index 1, image star index 2): image angular distance}
# - catalog_angular_distances: dict where {(HIP ID 1, HIP ID 2): catalog angular distance}
# - tolerance: angular distance tolerance
# - new_star: index of new image star being added to the assignment
def is_consistent(
    assignment, image_angular_distances, catalog_angular_distances, tolerance, new_star
):

    # Assignment doesn't have a pair of stars in it so is trivially consistent
    if len(assignment) < 2:
        return True

    # Iterate through all pairs of stars and their angular distances
    for (i1, i2), img_angle in image_angular_distances.items():

        # If the pair doesn't contain the star we're adding to the assignment we don't need to check it
        if new_star not in (i1, i2):
            continue

        # Check if pair is assigned HIP IDs
        if i1 in assignment and i2 in assignment:
            hip1 = assignment[i1]
            hip2 = assignment[i2]

            catalog_angle = catalog_angular_distances.get(
                (hip1, hip2)
            ) or catalog_angular_distances.get((hip2, hip1))

            if catalog_angle is None:
                return False

            if abs(catalog_angle - img_angle) > tolerance:
                return False
    return True


# Function that scores a solution based on how well it describes observed angular distances in the image
# Inputs:
# - solution: dict where {star image ID: HIP ID}
# - image_angular_distances: dict where {(image star index 1, image star index 2): image angular distance}
# - catalog_angular_distances: dict where {(HIP ID 1, HIP ID 2): catalog angular distance}
# - tolerance: angular distance tolerance
# Outputs:
# - score: a float that describes the current solution's score. The perfect solution would have a score of 0
def score_solution(
    solution, image_angular_distances, catalog_angular_distances, tolerance=TOLERANCE
):
    # Sum of total differences between image angular distance and catalog angular distance
    total_diff = 0
    # Count of stars that passed tolerance check
    count = 0

    for (s1, s2), img_ang_dist in image_angular_distances.items():

        if s1 in solution and s2 in solution:
            hip1, hip2 = solution[s1], solution[s2]

            cat_ang_dist = catalog_angular_distances.get(
                (hip1, hip2)
            ) or catalog_angular_distances.get((hip2, hip1))

            if cat_ang_dist is None:
                continue

            diff = abs(cat_ang_dist - img_ang_dist)
            # Account for cyclic warping to get correct diff
            diff = min(diff, 360 - diff)
            if diff > tolerance:
                continue
            total_diff += diff
            count += 1

    if count == 0:
        return float("inf")

    # coverage - the fraction of image stars mapped
    coverage = count / len(image_angular_distances)
    score = (total_diff / count) * (1 / coverage)
    return score


# Function that scores all solutions
# Inputs:
# - solutions: array of possible solution dicts
# - image_angular_distances: dict where {(image star index 1, image star index 2): image angular distance}
# - catalog_angular_distances: dict where {(HIP ID 1, HIP ID 2): catalog angular distance}
# Outputs:
# - scored_solutions: array of tuples where the first element of each tuple is the solution
# and the second is it's score
def load_solution_scoring(
    solutions, image_angular_distances, catalog_angular_distances
):
    return [
        (sol, score_solution(sol, image_angular_distances, catalog_angular_distances))
        for sol in solutions
    ]


def image_vector_matrix(image_unit_vectors):
    return np.array(image_unit_vectors, dtype=np.float64)


def catalog_vector_matrix(final_solution, catalog_unit_vectors):
    matrix = []
    for star, hip in final_solution[0].items():
        cat_vector = catalog_unit_vectors.get(hip)
        if cat_vector is None:
            raise ValueError(f"Catalog vector for HIP {hip} not found.")
        matrix.append(cat_vector)
    return np.array(matrix, dtype=np.float64)


# Function that approximates the rotational relationship between the two sets of vectors (image and catalog)
# in a matrix using the weighted sum of their outer products (You can represent any
# full-rank matrix as a sum of outer products of vectors)
# Inputs:
# - image_vectors: np array of the unit vectors from the image.
# - catalog_vectors: np array of the unit vectors from the catalog
# - weights: array of floats. Represent how much we can trust each pair of image to catalog vectors
# Outputs:
# - attitude_profile_matrix: 3x3 matrix that is the sum of the outer product of each pair of vectors,
# multiplied by the pair's weight (attitude profile matrix)
def build_attitude_profile_matrix(image_vectors, catalog_vectors, weights=None):

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


# Function that transforms Wahba's problem into quaternion form
# I don't fully understand the math behind why this matrix is constructed exactly in this way,
# but I got the end construction from the end of this lecture:
# https://www.youtube.com/watch?v=BL4KLCEBUUs
# Inputs:
# - attitude_profile_matrix: 3x3 matrix
# Outputs:
# - quaternion_attitude_profile_matrix: 4x4 matrix
def build_quaternion_attitude_profile_matrix(attitude_profile_matrix):
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


# Wahba's problem can be rewritten as maximizing this form: q^TKq
# where K is the quaternion_attitude_profile_matrix
# We solve this by finding the eigenvector with the maximum
# eigenvalue which then will correspond to the quaternion
# that best aligns with our rotation
# Inputs:
# - quaternion_attitude_profile_matrix: 4x4 matrix
# Outputs:
# - optimal_quaternion: quaternion that corresponds to our rotation [qw, qx, qy, qz]
def find_optimal_quaternion(quaternion_attitude_profile_matrix):
    eigenvalues, eigenvectors = np.linalg.eigh(quaternion_attitude_profile_matrix)
    max_index = np.argmax(eigenvalues)
    optimal_quaternion = eigenvectors[:, max_index]
    optimal_quaternion /= np.linalg.norm(optimal_quaternion)
    return optimal_quaternion


def compute_attitude_quaternion(image_vectors, catalog_vectors, weights=None):
    attitude_profile = build_attitude_profile_matrix(
        image_vectors, catalog_vectors, weights
    )
    quaternion_attitude_profile_matrix = build_quaternion_attitude_profile_matrix(
        attitude_profile
    )
    q = find_optimal_quaternion(quaternion_attitude_profile_matrix)

    return q


# Function to apply inverse rotation of quaternion to our catalog vectors. We should
# get vectors in our camera frame
# Inputs:
# - quaternion: rotation we got from find_optimal_quaternion()
# - catalog_vectors: the unit vectors of the assigned HIP IDs stars
# Output:
# - camera_frame_vectors: vectors in camera frame
def inverse_rotate_vectors(quaternion, catalog_vectors):
    rot_object = rotate.from_quat(quaternion)
    camera_frame_vectors = rot_object.inv().apply(catalog_vectors)
    return camera_frame_vectors


# Function to project vectors in our camera frame to pixel coordinates in the
# image plane
# Inputs:
# - vectors: array of vectors to project
# - params: array of camera params:
#   - params[0] = (f_x, f_y)
#   - params[1] = (c_x, c_y)
# Outputs:
# - projected_coords: coordinates of each point. If point is behind camera
# the element in the array is None
def project_to_image_plane(vectors, params):
    f_x, f_y = params[0]
    c_x, c_y = params[1]
    projected_coords = []
    for vector in vectors:
        x, y, z = vector
        if z <= 0:
            projected_coords.append(None)
        else:
            projected_x = f_x * (x / z) + c_x
            projected_y = f_y * (y / z) + c_y
            projected_coords.append((projected_x, projected_y))

    return projected_coords


# Function that reprojects the assigned catalog unit vectors back into
# our image plane
def reproject_vectors(quaternion, catalog_vectors, params):
    camera_vectors = inverse_rotate_vectors(quaternion, catalog_vectors)
    projected_coords = project_to_image_plane(camera_vectors, params)
    return projected_coords

# Function to calculate error (offset of image star coords to reprojected
# star coords)
def calculate_error(image_star_coords, reprojected_star_coords):
    error_rate = []
    for i in range(len(reprojected_star_coords)):
        x_i, y_i = image_star_coords[i]

        if reprojected_star_coords[i] is None:
            error_rate.append(float("inf"))
            continue

        x_p, y_p = reprojected_star_coords[i]

        d_x = (x_i - x_p) ** 2
        d_y = (y_i - y_p) ** 2
        dist = np.sqrt(d_x + d_y)

        error_rate.append(dist)

    return error_rate


def calculate_weights(error_rates):
    weights = []
    for error in error_rates:
        if error == float("inf"):
            weights.append(0)
        else:
            weights.append(1 / (error**2 + EPSILON))

    return weights


def lost_in_space():

    # star_coords = ip.find_brightest_stars(IMAGE_FILE, NUM_STARS)
    star_coords = ip.find_stars_with_advanced_filters(IMAGE_FILE, NUM_STARS)
    img_unit_vectors = star_coords_to_unit_vector(
        star_coords, (CENTER_X, CENTER_Y), FOCAL_LENGTH_X, FOCAL_LENGTH_Y
    )
    ang_dists = get_angular_distances(
        star_coords, (CENTER_X, CENTER_Y), FOCAL_LENGTH_X, FOCAL_LENGTH_Y
    )
    print(f"Angular distances:")
    for (i, j), ang_dist in ang_dists.items():
         print(f"{i}->{j}: {ang_dist}")

    all_catalog_angular_distances = load_catalog_angular_distances()
    hypotheses = load_hypotheses(ang_dists, all_catalog_angular_distances, TOLERANCE)
    sorted_hypothesises = dict(
        sorted(hypotheses.items(), key=lambda item: len(item[1]))
    )
    # for hypotheses in sorted_hypothesises.items():
    #     print(f"{hypotheses}")
    #     print("\n")

    assignment = {}
    image_stars = []

    for i in range(0, NUM_STARS):
        image_stars.append(i)

    solutions = []
    DFS(
        assignment,
        image_stars,
        sorted_hypothesises,
        ang_dists,
        all_catalog_angular_distances,
        TOLERANCE,
        solutions
    )
    
    # DFS_optimized(
    #     assignment,
    #     image_stars,
    #     sorted_hypothesises,
    #     ang_dists,
    #     all_catalog_angular_distances,
    #     TOLERANCE,
    #     solutions,
    #     time.time()
    # )

    scored_solutions = load_solution_scoring(
        solutions, ang_dists, all_catalog_angular_distances
    )
    for sol in scored_solutions:
        print(f"{sol}")

    sorted_arr = sorted(scored_solutions, key=lambda x: x[1])
    best_match = sorted_arr[0]
    best_match_solution = best_match[0]

    print(f"Best match: ")
    print(f"{best_match_solution}")

    cat_unit_vectors = load_catalog_unit_vectors("star_distances_sorted.db")

    mapped_image_vectors = []
    for key in best_match_solution.keys():
        mapped_image_vectors.append(img_unit_vectors[key])
    img_matrix = image_vector_matrix(mapped_image_vectors)
    cat_matrix = catalog_vector_matrix(best_match, cat_unit_vectors)

    quaternion = compute_attitude_quaternion(img_matrix, cat_matrix)
    # q_scalar_last = np.roll(quaternion, -1)
    # reprojected_coords = reproject_vectors(
    #     q_scalar_last,
    #     cat_matrix,
    #     [(FOCAL_LENGTH_X, FOCAL_LENGTH_Y), (CENTER_X, CENTER_Y)],
    # )
    
    # print(f"Reprojected:")
    # for el in reprojected_coords:
    #     if el is None:
    #         print(f"aaa")
    #     else:
    #         x, y = el
    #         print(f"{x}, {y}")

    # print(f"Original:")
    # for x, y in star_coords:
    #     print(f"{x}, {y}")
    
    # error_rates = calculate_error(star_coords, reprojected_coords)
    # for error in error_rates:
    #     print(f"{error}")

    # weights = calculate_weights(error_rates)
    # new_q = compute_attitude_quaternion(img_matrix, cat_matrix, weights)
    # print(f"New quaternion: {new_q}")

    ip.display_star_detections(IMAGE_FILE, star_coords)
    return quaternion


if __name__ == "__main__":
    q = lost_in_space()
    print(f"{q}")
