import sys
import os
import time
import math
import sqlite3
from itertools import combinations
from collections import defaultdict
from numpy.typing import ArrayLike, NDArray
from typing import Sequence, Tuple, Dict
from scipy.spatial.transform import Rotation as rotate
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import numpy as np
import pickle
import cProfile
import pstats

image_processing_folder_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "image_processing")
)
sys.path.append(image_processing_folder_path)
import image_processing_v6 as ip

IMAGE_HEIGHT = 1079  # 768
IMAGE_WIDTH = 1919  # 1366
ASPECT_RATIO = IMAGE_WIDTH / IMAGE_HEIGHT
FOV_Y = 53
FOV_X = math.degrees(2 * math.atan(math.tan(math.radians(FOV_Y / 2)) * ASPECT_RATIO))
CENTER_X = IMAGE_WIDTH / 2
CENTER_Y = IMAGE_HEIGHT / 2
FOCAL_LENGTH_X = (IMAGE_WIDTH / 2) / math.tan(math.radians(FOV_X / 2))
FOCAL_LENGTH_Y = (IMAGE_HEIGHT / 2) / math.tan(math.radians(FOV_Y / 2))
TOLERANCE = 2
IMAGE_FILE = "./test_images/testing119.png"
IMAGE_FILE2 = "./test_images/testing120.png"
IMAGE_FILE3 = "./test_images/testing121.png"
TEST_IMAGE = "./test_images/testing90.png"
OUT_FILE = "./out.prof"
NUM_STARS = 15
EPSILON = 1e-3
MIN_SUPPORT = 5
MIN_MATCHES = 10
MIN_VOTES_THRESHOLD = 2
MIN_MATCHES_TRACKING = 4
MAX_REFINEMENT_ITERATIONS = 20
QUATERNION_ANGLE_DIFFERENCE_THRESHOLD = 1e-3


def star_coords_to_unit_vector(
    star_coords: list[tuple[float, float]],
    center_coords: tuple[float, float],
    f_x: float,
    f_y: float,
) -> NDArray[np.float64]:
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
    if f_x == 0 or f_y == 0:
        raise ValueError("f_x and f_y must be non-zero.")

    if len(center_coords) != 2:
        raise ValueError("center_coords must be a tuple of (xc, yc)")

    star_coords_matrix = np.array(star_coords, dtype=float)

    if star_coords_matrix.size == 0:
        raise ValueError("star_coords must contain at least one (x, y) pair")

    if star_coords_matrix.ndim != 2 or star_coords_matrix.shape[1] != 2:
        raise ValueError("star_coords must be an array of (x, y) pairs")

    c_x, c_y = center_coords
    unit_vectors = []

    # Convert pixel coordinates into normalized camera coordinates (back-projection step)
    x_norm = (star_coords_matrix[:, 0] - c_x) / f_x
    y_norm = (star_coords_matrix[:, 1] - c_y) / f_y
    z_coord = np.ones_like(x_norm)

    vectors = np.stack([x_norm, y_norm, z_coord], axis=1)

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Encountered zero-length direction vector")

    unit_vectors = vectors / norms

    if not np.all(np.isfinite(unit_vectors)):
        raise ValueError("Non-finite values found in unit vectors")

    return unit_vectors


def angular_dist_helper(
    unit_vectors: NDArray[np.float64],
) -> dict[tuple[int, int], float]:
    """
    Angular distance between each unique pair of unit vectors
    Inputs:
    - unit_vectors: np array containing [x, y, z] coordinates of each unit vector
    Outputs:
    - angular_dists: dict where {(star_index1 [-], star_index2 [-]): angular_distance [deg]}
    """
    uv = np.asarray(unit_vectors, dtype=float)
    if uv.size == 0:
        return {}
    if uv.ndim != 2 or uv.shape[1] != 3:
        raise ValueError("unit_vectors must have shape (N, 3)")
    if not np.all(np.isfinite(uv)):
        raise ValueError("Non-finite values in unit vectors")

    # Multiply unit vector matrix by it's transpose to find the dot products
    dot_products = uv @ uv.T
    dot_products = np.clip(dot_products, -1.0, 1.0)

    angular_dist_matrix = np.degrees(np.arccos(dot_products))
    angular_dists = {
        (i, j): angular_dist_matrix[i, j] for i, j in combinations(range(len(uv)), 2)
    }
    return angular_dists


def get_angular_distances(
    star_coords: Sequence[tuple[float, float]],
    center_coords: tuple[float, float],
    f_x: float,
    f_y: float,
) -> dict[tuple[int, int], float]:
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
    conn = None
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = f"SELECT hip1, hip2, angular_distance FROM {table_name}"
    cursor.execute(query)
    rows = cursor.fetchall()

    conn.close()

    return {(row[0], row[1]): row[2] for row in rows}


def filter_catalog_angular_distances(
    cat_ang_dists: dict[tuple[int, int], float], bounds: tuple[float, float]
) -> dict[tuple[int, int], float]:
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


def generate_raw_votes(
    angular_distances: dict[tuple[int, int], float],
    catalog_hash: Dict[int, list[Tuple[int, int]]],
    tolerance: float,
) -> defaultdict[Tuple[int, int], int]:
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

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

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


def get_bounds(ang_dist: float, tolerance: float) -> tuple[float, float]:
    """
    Function that returns an angular distance interval based on initial angular distance and a tolerance
    Outputs:
    - bounds: tuple where bounds[0] = min_ang_dist and bounds[1] = max_ang_dist
    """
    return (ang_dist - tolerance, ang_dist + tolerance)


def build_hypotheses_from_votes(
    raw_votes: dict[Tuple[int, int], int], min_votes=1
) -> dict[int, list[int]]:
    """
    Build hypotheses dict for DFS based on raw votes from generate_raw_votes function
    """
    if not isinstance(raw_votes, dict):
        raise TypeError("raw_votes must be a dictionary")

    if not isinstance(min_votes, int):
        raise TypeError("min_votes must be an integer")

    hypotheses = defaultdict(set)
    for key, vote_count in raw_votes.items():
        if not isinstance(key, tuple) or len(key) != 2:
            raise ValueError(
                f"raw_votes key must be a tuple of (star_idx, hip_id), got {key}"
            )
        if not isinstance(vote_count, int):
            raise ValueError(f"vote count must be an integer, got {vote_count}")

        star_idx, hip_id = key
        if vote_count >= min_votes:
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

        for (hip1, hip2), _ in cur_cat_dict.items():
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
    min_matches,
    start_time,
    max_time=15,
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
    - min_matches: minimum required matches to be constituted as an assignment
    - start_time: time of start of execution
    """

    if time.time() - start_time >= max_time:
        return

    if len(assignment) >= min_matches:
        solutions.append(dict(assignment))

    if len(assignment) == len(image_stars):
        return

    unassigned = [s for s in image_stars if s not in assignment]
    
    if unassigned is None:
        return
    
    current_star = min(unassigned, key=lambda s: len(candidate_hips.get(s, [])))

    for hip_candidate in candidate_hips.get(current_star, []):

        if hip_candidate in assignment.values():
            continue

        assignment[current_star] = hip_candidate

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
                min_matches,
                start_time,
                max_time,
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

    if len(assignment) < 2:
        return True

    for (i1, i2), img_angle in image_angular_distances.items():

        if new_star not in (i1, i2):
            continue

        if i1 in assignment and i2 in assignment:
            hip1 = assignment[i1]
            hip2 = assignment[i2]

            catalog_angle = catalog_angular_distances.get((hip1, hip2))
            if catalog_angle is None:
                catalog_angle = catalog_angular_distances.get((hip2, hip1))
                
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

    with open("catalog_hash.pkl", "rb") as f:
        catalog_hash = pickle.load(f)
    all_catalog_angular_distances = load_catalog_angular_distances()

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
    
    ip.display_star_detections(image_file, star_coords)

    # hypotheses = load_hypotheses(ang_dists, all_catalog_angular_distances, TOLERANCE)
    raw_votes = generate_raw_votes(ang_dists, catalog_hash, TOLERANCE)
    if not raw_votes:
        print("No raw votes generated")
        return None

    hypotheses = build_hypotheses_from_votes(raw_votes, MIN_VOTES_THRESHOLD)
    if not hypotheses:
        print("No hypotheses generated")
        return None

    current_hypotheses = hypotheses
    solutions = []

    while len(current_hypotheses) >= MIN_MATCHES:
        image_stars_to_try = list(current_hypotheses.keys())
        sorted_hypotheses = dict(
            sorted(current_hypotheses.items(), key=lambda item: len(item[1]))
        )
        print(f"  Attempting to solve with {len(image_stars_to_try)} stars...")
        iteration_solutions = []
        assignment = {}
        dfs_start_time = time.time()

        DFS(
            assignment,
            image_stars_to_try,
            sorted_hypotheses,
            ang_dists,
            all_catalog_angular_distances,
            TOLERANCE,
            iteration_solutions,
            MIN_MATCHES,
            dfs_start_time,
            max_time=15,
        )

        if iteration_solutions:
            print(
                f"  --> SUCCESS: Found a consistent group in {time.time() - dfs_start_time:.2f} seconds!"
            )
            solutions = iteration_solutions
            break
        else:
            print(f"  --> FAILED. Assuming an outlier is present in the current set.")
            # Identify the least reliable star. The stars with the most votes are usually the brightest ones.
            # So this starts deleting star 0,1,2.... Which might delete a real star, but it shouldnt be a problem
            if not current_hypotheses:
                break
            star_to_remove = max(
                current_hypotheses.items(), key=lambda item: len(item[1])
            )[0]
            print(f"Removing Star {star_to_remove} (most ambiguous) and trying again.")

            del current_hypotheses[star_to_remove]

    if not solutions:
        print(
            "\n!!! FAILED: Could not find any geometrically consistent solution even after removing outliers."
        )
        return None

    scored_solutions = load_solution_scoring(
        solutions, ang_dists, all_catalog_angular_distances
    )

    sorted_arr = sorted(scored_solutions, key=lambda x: (-len(x[0]), x[1]))
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

    for _ in range(0, MAX_REFINEMENT_ITERATIONS):
        refined_quaternion = refine_quaternion(final_quaternion, cat_matrix, img_matrix)
        delta_angle = rotational_angle_between_quaternions(
            final_quaternion, refined_quaternion
        )
        if delta_angle <= QUATERNION_ANGLE_DIFFERENCE_THRESHOLD:
            break
        final_quaternion = refined_quaternion
        
    used_coords = []
    for key in best_match_solution.keys():
        used_coords.append(star_coords[key])
    return final_quaternion, cat_matrix, used_coords, best_match_solution


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


def match_stars_hungarian(
    predicted_positions, detected_positions, distance_threshold=10.0
):

    if (
        predicted_positions is None
        or len(predicted_positions) == 0
        or detected_positions is None
        or len(detected_positions) == 0
    ):
        return []

    valid_predicted = []
    valid_detected = []
    
    for index, prev_coords in enumerate(predicted_positions):
        if (prev_coords is not None and 
            len(prev_coords) == 2 and 
            all(np.isfinite(coord) for coord in prev_coords)):
            valid_predicted.append((index, prev_coords))
    
    for index, detected_coords in enumerate(detected_positions):
        if (detected_coords is not None and 
            len(detected_coords) == 2 and 
            all(np.isfinite(coord) for coord in detected_coords)):
            valid_detected.append((index, detected_coords))

    if not valid_predicted or not valid_detected:
        return []

    predicted_indices, predicted_coords = zip(*valid_predicted)
    detected_indices, detected_coords = zip(*valid_detected)
    predicted_array = np.array(predicted_coords)
    detected_array = np.array(detected_coords)

    distances_matrix = cdist(predicted_array, detected_array)
    
    cost_matrix = np.where(
        distances_matrix <= distance_threshold, distances_matrix, 1e6
    )

    hungarian_pred_indices, hungarian_det_indices = linear_sum_assignment(cost_matrix)
    
    matches = []
    for filtered_pred_idx, filtered_det_idx in zip(hungarian_pred_indices, hungarian_det_indices):
        if cost_matrix[filtered_pred_idx, filtered_det_idx] < 1e6:
            original_pred_idx = predicted_indices[filtered_pred_idx]
            original_det_idx = detected_indices[filtered_det_idx]
            matches.append((original_pred_idx, original_det_idx))

    return matches


def track(
    previous_quaternion,
    previous_catalog_unit_vectors,
    previous_star_coords,
    previous_mapping,
    detected_star_coords,
    camera_params,
    min_matches,
    distance_threshold=10.0
):
    if previous_quaternion is None or len(detected_star_coords) < min_matches:
        return previous_quaternion, [], [], [], {}

    original_star_indices = list(previous_mapping.keys())
    original_star_indices.sort()
    
    sequential_to_original = {i: original_star_indices[i] for i in range(len(original_star_indices))}

    matches = match_stars_hungarian(
        previous_star_coords, detected_star_coords, distance_threshold
    )
    
    if len(matches) < min_matches:
        print("Not enough matches to track")
        return previous_quaternion, matches, [], [], {}

    matched_detected_pixels = [detected_star_coords[det_idx] for _, det_idx in matches]
    
    image_vectors = star_coords_to_unit_vector(
        matched_detected_pixels,
        center_coords=camera_params[1],
        f_x=camera_params[0][0],
        f_y=camera_params[0][1],
    )

    catalog_vectors = np.array([previous_catalog_unit_vectors[pred_idx] for pred_idx, _ in matches])
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
    
    used_catalog_vectors = [previous_catalog_unit_vectors[pred_idx] for pred_idx, _ in matches]
    used_star_coords = [detected_star_coords[det_idx] for _, det_idx in matches]
    
    tracking_solution = {}
    for pred_idx, det_idx in matches:
        if pred_idx in sequential_to_original:
            original_star_idx = sequential_to_original[pred_idx]
            star_id = previous_mapping[original_star_idx]
            tracking_solution[det_idx] = star_id
            
    return final_quaternion, matches, used_catalog_vectors, used_star_coords, tracking_solution


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
    # profiler = cProfile.Profile()
    # profiler.enable()
    
    mode = "LIS"
    quaternion = None
    catalog_matrix = None
    coords = None
    best_solution = None

    image_files = ["./test_images/testing142.png"]

    for idx, img in enumerate(image_files):
        print(f"\n--- Processing image {idx+1} ---")
    
        if mode == "LIS":
            begin_time = time.time()
            lost_in_space_element = lost_in_space(img)
            if lost_in_space_element is None:
                print("No lost in space solution")
                continue
            quaternion, catalog_matrix, coords, best_solution = lost_in_space_element
            end_time = time.time()
            print(f"LIS quaternion: {quaternion}")
            print(f"LIS time: {end_time - begin_time:.3f}s")
            print(f"Catalog matrix shape: {catalog_matrix.shape}")
        
            if quaternion is not None:
                mode = "TRACKING"
                print("Switched to TRACKING mode")
            else:
                print("LIS failed to determine attitude!")
                continue

        elif mode == "TRACKING":
            detected_star_coords = ip.find_stars_with_advanced_filters(img, NUM_STARS)
            ip.display_star_detections(img, detected_star_coords, f"stars_identified_{idx+1}.png")

            begin_time = time.time()
            new_quaternion, matches, tracking_catalog_vectors, tracking_star_coords, tracking_solution = track(
                quaternion,
                catalog_matrix,
                coords,
                best_solution,
                detected_star_coords,
                [(FOCAL_LENGTH_X, FOCAL_LENGTH_Y), (CENTER_X, CENTER_Y)],
                MIN_MATCHES_TRACKING,
                distance_threshold=140.0,
            )
            end_time = time.time()
        
            print(f"Tracking time: {end_time - begin_time:.3f}s")
            print(f"Tracking quaternion: {new_quaternion}")
            print(f"Matches: {matches}")
        
            if len(matches) < MIN_MATCHES_TRACKING:
                print("Tracking failed → switching to LOST-IN-SPACE")
                mode = "LIS"
            else:
                print(f"Rotational angle: {rotational_angle_between_quaternions(quaternion, new_quaternion)}")
                quaternion = new_quaternion
                catalog_matrix = np.array(tracking_catalog_vectors)
                coords = tracking_star_coords
                best_solution = tracking_solution
#     profiler.disable()
#     profiler.dump_stats(OUT_FILE)
#     stats = pstats.Stats(OUT_FILE).strip_dirs().sort_stats("cumulative")
# stats.print_stats(5)
