import sys
import os
import time
import math
import sqlite3
from itertools import combinations
from collections import defaultdict
from scipy.spatial.transform import Rotation as rotate
from scipy.spatial.distance import cdist
import numpy as np

image_processing_folder_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "image_processing")
)
sys.path.append(image_processing_folder_path)
import image_processing_v6 as ip

IMAGE_HEIGHT = 1964  # 768
IMAGE_WIDTH = 3024  # 1366
ASPECT_RATIO = IMAGE_WIDTH / IMAGE_HEIGHT
FOV_Y = 53
FOV_X = math.degrees(2 * math.atan(math.tan(math.radians(FOV_Y / 2)) * ASPECT_RATIO))
CENTER_X = IMAGE_WIDTH / 2
CENTER_Y = IMAGE_HEIGHT / 2
FOCAL_LENGTH_X = (IMAGE_WIDTH / 2) / math.tan(math.radians(FOV_X / 2))
FOCAL_LENGTH_Y = (IMAGE_HEIGHT / 2) / math.tan(math.radians(FOV_Y / 2))
TOLERANCE = 3
IMAGE_FILE = "./test_images/testing91.png"
IMAGE_FILE2 = "./test_images/testing92.png"
TEST_IMAGE = "./test_images/testing90.png"
NUM_STARS = 15
EPSILON = 1e-3
MIN_SUPPORT = 5
MIN_MATCHES = 5
MIN_VOTES_THRESHOLD = 2
MAX_REFINEMENT_ITERATIONS = 20
QUATERNION_ANGLE_DIFFERENCE_THRESHOLD = 1e-3


def star_coords_to_unit_vector(star_coords, center_coords, f_x, f_y):
    """
    Unit vector function -> finds star unit vectros based on star pixel coordinates
    using pinhole projection
    We're only finding the direction where the 3D object lies since we don't know the actual depth.
    Inputs:
    - star_coords: array of tuples that holds (x, y) pixel coordinates of stars in the image
    - center_coords: image center (principal point) coordinates in tuple form: (xc, yc)
    - f_x: const. Describes the scaling factor by x
    - f_y: const. Describes the scaling factor by y
    Outputs:
    - unit_vectors: array of np arrays holding (x, y, z) coordinates of each unit vector
    """
    # Unpack center_coords
    c_x, c_y = center_coords

    star_coords_matrix = np.array(star_coords)
    # Initialize empty unit vector array
    unit_vectors = []

    # Reverse pinhole projection to get X, Y, Z coords of each star

    # Convert pixel coordinates into normalized camera coordinates (back-projection step)
    # These correspond to where the pixel projects onto the image plane at depth z_p
    # Inversing pinhole projection formula for x (back-projecting). Formula: x = f_x * (x_p / z_p) + c_x
    x_norm = (star_coords_matrix[:, 0] - c_x) / f_x
    # Inversing pinhole projection formula for y (back-projecting). Formula: y = f_y * (x_p / z_p) + c_y
    y_norm = (star_coords_matrix[:, 1] - c_y) / f_y
    # Pick arbitrary depth for each 3D point
    # (we don't know how far away they really are, we just want their direction)
    z_coord = np.ones_like(x_norm)

    # array holding [X, Y, Z] coordinates of the direction vector
    vectors = np.stack([x_norm, y_norm, z_coord], axis=1)

    # Normalize vector to save on calculations later on
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit_vectors = vectors / norms

    return unit_vectors


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


def load_catalog_angular_distances(
    db_path="star_distances_sorted.db", table_name="AngularDistances"
):
    """
    Function that loads a dict from the database where {(HIP ID 1 [-], HIP ID 2 [-]): angular_distance [deg]}
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = f"SELECT hip1, hip2, angular_distance FROM {table_name}"
    cursor.execute(query)
    rows = cursor.fetchall()

    conn.close()

    return {(row[0], row[1]): row[2] for row in rows}


def filter_catalog_angular_distances(cat_ang_dists, bounds):
    """
    Function that returns a dict where {(HIP ID 1 [-], HIP ID 2 [-]): angular_distance [deg]}
    angular_distance ∈ [min_ang_dist, max_ang_dist]
    Inputs:
    - cat_ang_dists: dict where {(HIP ID 1 [-], HIP ID 2 [-]): angular_distance [deg]}
    - bounds[2]: array where bounds[0] = min_ang_dist and bounds[1] = max_ang_dist
    """
    min_ang_dist, max_ang_dist = bounds
    filtered = {
        pair: ang_dist
        for pair, ang_dist in cat_ang_dists.items()
        if min_ang_dist <= ang_dist <= max_ang_dist
    }
    return filtered


def generate_raw_votes(angular_distances, catalog_hash, tolerance):
    """
    Function that returns initial votes for each star index - hip id
    mapping based on the catalog hash pairs for our current bin
    Inputs:
    - angular_distances
    - catalog_hash: dict of pairs {bin number: [candidate pairs]}
    - tolerance: angular distance tolerance
    Outputs:
    - votes: dict {(image star index, HIP ID): number of votes}
    """

    votes = defaultdict(int)

    for (s1, s2), ang_dist in angular_distances.items():
        if ang_dist is None:
            continue
        key = int(ang_dist / tolerance)
        candidate_pairs = catalog_hash.get(key, [])
        for hip1, hip2 in candidate_pairs:
            votes[(s1, hip1)] += 1
            votes[(s1, hip2)] += 1
            votes[(s2, hip1)] += 1
            votes[(s2, hip2)] += 1

    return votes


def get_bounds(ang_dist, tolerance):
    """
    Function that returns an angular distance interval based on initial angular distance and a tolerance
    Outputs:
    - bounds: tuple where bounds[0] = min_ang_dist and bounds[1] = max_ang_dist
    """
    return (ang_dist - tolerance, ang_dist + tolerance)


def build_hypotheses_from_votes(raw_votes, min_votes=1):
    """
    Build hypotheses dict for DFS based on raw votes from generate_raw_votes function
    """
    hypotheses = defaultdict(set)
    for (star_idx, hip_id), vote_count in raw_votes.items():
        if vote_count < min_votes:
            hypotheses[star_idx].add(hip_id)

    return hypotheses


def load_hypotheses(angular_distances, all_cat_ang_dists, tolerance):
    """
    Function that loads candidate catalog HIP IDs for each image star
    Inputs:
    - num_stars: number of stars from the image
    - angular_distances: dict where {(image star index 1, image star index 2): image angular distance}
    - all_cat_ang_dists: dict where {(HIP ID 1, HIP ID 2): catalog angular distance}
    - tolerance: angular distance tolerance
    Outputs:
    - hypotheses_dict: {image star index: [HIP ID 1, ... HIP ID N]}
    """
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


def DFS(
    assignment,
    image_stars,
    candidate_hips,
    image_angular_distances,
    catalog_angular_distances,
    tolerance,
    solutions,
    start_time,
    max_time=5,
):
    """
    DFS to produce every possible mapping of image star to catalog star ID (even if not full mappings)
    Inputs:
    - assignment: a dict mapping stars in the image to candidate catalog stars (HIP IDs)
    - image_stars: list of stars detected in the image
    - candidate_hips: dict mapping each image star to a list of candidate catalog HIP IDs
    - image_angular_distances: dict where {(image star index 1, image star index 2): image angular distance}
    - catalog_angular_distances: dict where {(HIP ID 1, HIP ID 2): catalog angular distance}
    - tolerance: angular distance tolerance
    - solutions: a list to collect valid assignments (solutions) - initially empty
    """

    # Does current assignment qualify as a solution
    if len(assignment) >= MIN_MATCHES:
        solutions.append(dict(assignment))

    # End of recursion condition if we've matched all stars
    if len(assignment) == len(image_stars):
        return

    if abs(start_time - time.time()) == max_time:
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
                start_time,
            )

        del assignment[current_star]


def is_consistent(
    assignment, image_angular_distances, catalog_angular_distances, tolerance, new_star
):
    """
    Bool function that checks if the current assignment is consistent with the angular distances from our image within a certain tolerance
    Inputs:
    - assignment: current assignment
    - image_angular_distances: dict where {(image star index 1, image star index 2): image angular distance}
    - catalog_angular_distances: dict where {(HIP ID 1, HIP ID 2): catalog angular distance}
    - tolerance: angular distance tolerance
    - new_star: index of new image star being added to the assignment
    """

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


def score_solution(
    solution, image_angular_distances, catalog_angular_distances, tolerance=TOLERANCE
):
    """
    Function that scores a solution based on how well it describes observed angular distances in the image
    Inputs:
    - solution: dict where {star image ID: HIP ID}
    - image_angular_distances: dict where {(image star index 1, image star index 2): image angular distance}
    - catalog_angular_distances: dict where {(HIP ID 1, HIP ID 2): catalog angular distance}
    - tolerance: angular distance tolerance
    Outputs:
    - score: a float that describes the current solution's score. The perfect solution would have a score of 0
    """

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


def load_solution_scoring(
    solutions, image_angular_distances, catalog_angular_distances
):
    """
    Function that scores all solutions
    Inputs:
    - solutions: array of possible solution dicts
    - image_angular_distances: dict where {(image star index 1, image star index 2): image angular distance}
    - catalog_angular_distances: dict where {(HIP ID 1, HIP ID 2): catalog angular distance}
    Outputs:
    - scored_solutions: array of tuples where the first element of each tuple is the solution
    and the second is it's score
    """
    return [
        (sol, score_solution(sol, image_angular_distances, catalog_angular_distances))
        for sol in solutions
    ]


def image_vector_matrix(image_unit_vectors):
    return np.array(image_unit_vectors, dtype=np.float64)


def catalog_vector_matrix(final_solution, catalog_unit_vectors):
    matrix = []
    for _, hip in final_solution[0].items():
        cat_vector = catalog_unit_vectors.get(hip)
        if cat_vector is None:
            raise ValueError(f"Catalog vector for HIP {hip} not found.")
        matrix.append(cat_vector)
    return np.array(matrix, dtype=np.float64)


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


def inverse_rotate_vectors(quaternion, catalog_vectors):
    """
    Function to apply inverse rotation of quaternion to our catalog vectors. We should
    get vectors in our camera frame
    Inputs:
    - quaternion: rotation in [w, x, y, z] format
    - catalog_vectors: the unit vectors of the assigned HIP IDs stars
    Output:
    - camera_frame_vectors: vectors in camera frame
    """
    # Convert from [w, x, y, z] to [x, y, z, w] for scipy
    q_scalar_last = np.roll(quaternion, -1)
    rot_object = rotate.from_quat(q_scalar_last)
    camera_frame_vectors = rot_object.inv().apply(catalog_vectors)
    return camera_frame_vectors


def project_to_image_plane(vectors, params):
    """
    Function to project vectors in our camera frame to pixel coordinates in the
    image plane
    Inputs:
    - vectors: array of vectors to project
    - params: array of camera params:
      - params[0] = (f_x, f_y)
      - params[1] = (c_x, c_y)
    Outputs:
    - projected_coords: coordinates of each point. If point is behind camera
    the element in the array is None
    """
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


def reproject_vectors(quaternion, catalog_vectors, params):
    """
    Function that reprojects the assigned catalog unit vectors back into
    our image plane
    """
    camera_vectors = inverse_rotate_vectors(quaternion, catalog_vectors)
    projected_coords = project_to_image_plane(camera_vectors, params)
    return projected_coords


def calculate_error(image_star_coords, reprojected_star_coords):
    """
    Function to calculate error (offset of image star coords to reprojected
    star coords)
    """
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


def calculate_weights(error_rates):
    """
    Calculate weights of how much we trust each vector pair based on error rates (euqlidian distances
    between image star and reprojected stars)
    Inputs:
    - error_rates: euqlidian distances between image star and reprojected stars
    Outputs:
    - weights: array of floats - how much we trust each vector pair
    """
    weights = []
    for error in error_rates:
        if error == float("inf"):
            weights.append(0)
        else:
            weights.append(1 / (error**2 + EPSILON))

    return weights


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


def lost_in_space(image_file=None):
    if image_file == None:
        image_file = TEST_IMAGE

    # star_coords = ip.find_brightest_stars(IMAGE_FILE, NUM_STARS)
    star_coords = ip.find_stars_with_advanced_filters(image_file, NUM_STARS)
    img_unit_vectors = star_coords_to_unit_vector(
        star_coords, (CENTER_X, CENTER_Y), FOCAL_LENGTH_X, FOCAL_LENGTH_Y
    )
    ang_dists = get_angular_distances(
        star_coords, (CENTER_X, CENTER_Y), FOCAL_LENGTH_X, FOCAL_LENGTH_Y
    )
    # print("Angular distances:")
    # for (i, j), ang_dist in ang_dists.items():
    #     print(f"{i}->{j}: {ang_dist}")

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
        solutions,
        time.time(),
    )

    scored_solutions = load_solution_scoring(
        solutions, ang_dists, all_catalog_angular_distances
    )
    # for sol in scored_solutions:
    #     print(f"{sol}")

    sorted_arr = sorted(scored_solutions, key=lambda x: x[1])
    best_match = sorted_arr[0]
    best_match_solution = best_match[0]

    print("Best match: ")
    print(f"{best_match_solution}")

    cat_unit_vectors = load_catalog_unit_vectors("star_distances_sorted.db")

    mapped_image_vectors = []
    for key in best_match_solution.keys():
        mapped_image_vectors.append(img_unit_vectors[key])
    img_matrix = image_vector_matrix(mapped_image_vectors)
    cat_matrix = catalog_vector_matrix(best_match, cat_unit_vectors)

    quaternion = compute_attitude_quaternion(img_matrix, cat_matrix)
    final_quaternion = quaternion

    for i in range(0, MAX_REFINEMENT_ITERATIONS):
        refined_quaternion = refine_quaternion(final_quaternion, cat_matrix, img_matrix)
        delta_angle = rotational_angle_between_quaternions(
            final_quaternion, refined_quaternion
        )
        if delta_angle <= QUATERNION_ANGLE_DIFFERENCE_THRESHOLD:
            break
        final_quaternion = refined_quaternion

    ip.display_star_detections(image_file, star_coords)
    return final_quaternion, cat_matrix, star_coords


def match_stars(predicted_positions, detected_positions, distance_threshold=10.0):
    """
    Match predicted star positions to detected star positions using Euclidean distance.
    Inputs:
    - predicted_positions (list of (x, y)): Projected star coords from previous attitude.
    - detected_positions (list of (x, y)): Star coords detected in current image.
    - distance_threshold (float): Max pixel distance for matching.
    Outputs:
    - matches: list of tuples (predicted_idx, detected_idx) of matched pairs.
    """

    valid_predicted = [
        (index, prev_coords)
        for index, prev_coords in enumerate(predicted_positions)
        if prev_coords is not None and len(prev_coords) == 2
    ]
    valid_detected = [
        (index, detected_coords)
        for index, detected_coords in enumerate(detected_positions)
        if detected_coords is not None and len(detected_coords) == 2
    ]

    predicted_indices, predicted_coords = zip(*valid_predicted)
    detected_indices, detected_coords = zip(*valid_detected)
    predicted_array = np.array(predicted_coords)
    detected_array = np.array(detected_coords)

    distances_matrix = cdist(predicted_array, detected_array)
    distances_matrix[distances_matrix > distance_threshold] = np.inf

    matches = []
    used_detected = set()

    for i, row in enumerate(distances_matrix):
        min_index = np.argmin(row)
        if row[min_index] != np.inf and min_index not in used_detected:
            matches.append((predicted_indices[i], detected_indices[min_index]))
            used_detected.add(min_index)

    return matches


def track(
    previous_quaternion,
    previous_catalog_unit_vectors,
    previous_star_coords,
    detected_star_coords,
    camera_params,
    distance_threshold=10.0,
):
    """
    Perform star tracking by matching predicted star positions to detected stars,
    then update the attitude quaternion.
    Inputs:
    - previous_quaternion: Current attitude quaternion estimate in [w, x, y, z] format.
    - previous_catalog_unit_vectors: Star unit vectors from catalog for previous frame.
    - detected_star_coords: Detected star pixel coords in current image.
    - camera_params: ((f_x, f_y), (c_x, c_y)) camera intrinsic parameters.
    - distance_threshold: Pixel distance threshold for matching.
    Outputs:
    - updated_quaternion: New attitude quaternion.
    - matches: List of matched star index pairs.
    """

    matches = match_stars(
        previous_star_coords, detected_star_coords, distance_threshold
    )
    if len(matches) < 3:
        print("Not enough matches to track")
        return previous_quaternion, matches

    matched_detected_pixels = [detected_star_coords[det_idx] for _, det_idx in matches]

    # Convert matched detected pixels to unit vectors
    image_vectors = star_coords_to_unit_vector(
        matched_detected_pixels,
        center_coords=camera_params[1],
        f_x=camera_params[0][0],
        f_y=camera_params[0][1],
    )

    # Extract matched catalog unit vectors using the correct indexing
    # The matches contain (previous_idx, detected_idx), where previous_idx
    # corresponds to the index in previous_catalog_unit_vectors
    catalog_vectors = np.array(
        [previous_catalog_unit_vectors[pred_idx] for pred_idx, _ in matches]
    )

    updated_quaternion = compute_attitude_quaternion(image_vectors, catalog_vectors)
    final_quaternion = updated_quaternion

    for _ in range(0, MAX_REFINEMENT_ITERATIONS):
        refined_quaternion = refine_quaternion(
            final_quaternion, catalog_vectors, image_vectors
        )
        delta_angle = rotational_angle_between_quaternions(
            final_quaternion, refined_quaternion
        )
        if delta_angle <= QUATERNION_ANGLE_DIFFERENCE_THRESHOLD:
            break
        final_quaternion = refined_quaternion

    return updated_quaternion, matches


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


if __name__ == "__main__":
    begin_time = time.time()
    q, cat_matrix, coords = lost_in_space(IMAGE_FILE)
    end_time = time.time()
    print(f"Lost in space quaternion: {q}")
    print(f"Lost in space time: {end_time - begin_time}")
    print(f"Catalog matrix shape: {cat_matrix.shape}")

    new_coords = ip.find_stars_with_advanced_filters(IMAGE_FILE2, NUM_STARS)

    begin_time = time.time()
    new_q, matches = track(
        q,
        cat_matrix,
        coords,
        new_coords,
        [(FOCAL_LENGTH_X, FOCAL_LENGTH_Y), (CENTER_X, CENTER_Y)],
        distance_threshold=50.0,
    )
    end_time = time.time()

    print(f"Tracking time {end_time - begin_time}")
    print(f"Tracking quaternion: {new_q}")
    print(f"Matches: {matches}")

    if len(matches) == 0:
        print("\nTrying with even larger threshold...")
        new_q, matches = track(
            q,
            cat_matrix,
            new_coords,
            [(FOCAL_LENGTH_X, FOCAL_LENGTH_Y), (CENTER_X, CENTER_Y)],
            distance_threshold=100.0,
        )
        print(f"Matches with 100px threshold: {matches}")

    # new_q, c, sc = lost_in_space(IMAGE_FILE2)
    rotational_angle = rotational_angle_between_quaternions(q, new_q)
    print(f"{rotational_angle}")
