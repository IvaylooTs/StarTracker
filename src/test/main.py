import sys
import os
import time
import math
import sqlite3
import pickle
from itertools import combinations
from collections import defaultdict
from scipy.spatial.transform import Rotation as rotate
import numpy as np
image_processing_folder_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "image_processing")
)
sys.path.append(image_processing_folder_path)
import image_processing_v6 as ip

IMAGE_HEIGHT = 1080  # 1964
IMAGE_WIDTH = 1920  # 3024
ASPECT_RATIO = IMAGE_HEIGHT / IMAGE_WIDTH
FOV_Y = 52
FOV_X = math.degrees(
    2 * math.atan(math.tan(math.radians(FOV_Y / 2)) * (1 / ASPECT_RATIO))
)
CENTER_X = IMAGE_WIDTH / 2
CENTER_Y = IMAGE_HEIGHT / 2
FOCAL_LENGTH_X = (IMAGE_WIDTH / 2) / math.tan(math.radians(FOV_X / 2))
FOCAL_LENGTH_Y = (IMAGE_HEIGHT / 2) / math.tan(math.radians(FOV_Y / 2))
TOLERANCE = 1.5
IMAGE_FILE = "src\digital_twin\\test_images\\saturn2.png"
NUM_STARS = 15
EPSILON = 1e-6
MIN_SUPPORT = 1
MIN_MATCHES = 7

# Place this function in main.py
def get_initial_identities(all_votes):
    """
    Finds the best single HIP ID candidate for each image star based on raw vote counts.

    Args:
        all_votes: A dictionary from the hashing stage, like {(star_idx, hip_id): vote_count}

    Returns:
        A dictionary representing the "proposed solution", like {star_idx: best_hip_id}
    """
    star_candidates = defaultdict(list)
    for (star_idx, hip_id), count in all_votes.items():
        star_candidates[star_idx].append((hip_id, count))

    proposed_solution = {}
    for star_idx, candidates in star_candidates.items():
        # Sort candidates by vote count (descending) and pick the top one
        if candidates:
            best_hip_id, _ = max(candidates, key=lambda item: item[1])
            proposed_solution[star_idx] = best_hip_id
            
    return proposed_solution

# Place this function in main.py
def perform_validation_voting(proposed_solution, image_angular_distances, catalog_angular_distances, tolerance):
    """
    Performs the second, validation vote based on geometric consistency.

    Args:
        proposed_solution: The "best guess" map {image_star: hip_id}
        image_angular_distances: The measured distances from the image.
        catalog_angular_distances: The true distances from the catalog.
        tolerance: The tolerance for a distance match.

    Returns:
        A dictionary of validation vote counts, like {image_star: validation_vote_count}
    """
    validation_votes = defaultdict(int)
    image_star_indices = list(proposed_solution.keys())

    # Iterate through all pairs of stars in our proposed solution
    for i in range(len(image_star_indices)):
        for j in range(i + 1, len(image_star_indices)):
            s1 = image_star_indices[i]
            s2 = image_star_indices[j]

            # Get the measured distance from the image
            img_dist = image_angular_distances.get((s1, s2))
            if img_dist is None:
                continue

            # Get the corresponding HIP IDs from our proposed solution
            hip1 = proposed_solution[s1]
            hip2 = proposed_solution[s2]

            # Get the true distance from the catalog
            cat_dist = catalog_angular_distances.get((hip1, hip2)) or catalog_angular_distances.get((hip2, hip1))
            if cat_dist is None:
                continue
            
            # Check for geometric consistency
            if abs(img_dist - cat_dist) <= tolerance:
                # If they are consistent, they support each other. Cast validation votes.
                validation_votes[s1] += 1
                validation_votes[s2] += 1
                
    return validation_votes

def build_hypotheses_from_votes(raw_votes, min_votes=1):
    """
    Converts the raw vote counts into a hypothesis dictionary suitable for DFS.

    Args:
        raw_votes: The dict from generate_raw_votes, {(star_idx, hip_id): count}
        min_votes: The minimum number of votes for a pair to be a hypothesis.

    Returns:
        A dictionary like {star_idx: {hip_id1, hip_id2, ...}}
    """
    hypotheses = defaultdict(set)
    # To be more robust, let's only consider candidates with a reasonable number of votes.
    # This value might need tuning. If you get no solutions, try lowering it.
    MIN_VOTES_THRESHOLD = 1

    for (star_idx, hip_id), count in raw_votes.items():
        if count >= MIN_VOTES_THRESHOLD:
            hypotheses[star_idx].add(hip_id)
            
    return hypotheses

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
    db_path="src/digital_twin/star_distances_sorted.db", table_name="AngularDistances"
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


def get_bounds(ang_dist, tolerance):
    """
    Function that returns an angular distance interval based on initial angular distance and a tolerance
    Outputs:
    - bounds: tuple where bounds[0] = min_ang_dist and bounds[1] = max_ang_dist
    """
    return (ang_dist - tolerance, ang_dist + tolerance)


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
    if time.time() - start_time > max_time:
        return  # Stop searching if the time limit is exceeded
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
                start_time,  # Pass the original start_time
                max_time,  # Pass the max_time
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


def catalog_vector_matrix(final_solution_wrapper, catalog_unit_vectors):
    # The actual solution dict is now nested
    solution_dict = final_solution_wrapper['solution'] 
    matrix = []
    for _, hip in solution_dict.items():
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
    attitude_profile = build_attitude_profile_matrix(
        image_vectors, catalog_vectors, weights
    )
    quaternion_attitude_profile_matrix = build_quaternion_attitude_profile_matrix(
        attitude_profile
    )
    q = find_optimal_quaternion(quaternion_attitude_profile_matrix)

    return q


def inverse_rotate_vectors(quaternion, catalog_vectors):
    """
    Function to apply inverse rotation of quaternion to our catalog vectors. We should
    get vectors in our camera frame
    Inputs:
    - quaternion: rotation we got from find_optimal_quaternion()
    - catalog_vectors: the unit vectors of the assigned HIP IDs stars
    Output:
    - camera_frame_vectors: vectors in camera frame
    """
    rot_object = rotate.from_quat(quaternion)
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


def calculate_weights(error_rates):
    weights = []
    for error in error_rates:
        if error == float("inf"):
            weights.append(0)
        else:
            weights.append(1 / (error**2 + EPSILON))

    return weights


def generate_raw_votes(angular_distances, catalog_hash, tolerance):
    """Generates raw votes by looking up measured distances in the hash table."""
    votes = defaultdict(int)
    for (s1, s2), img_dist in angular_distances.items():
        if img_dist is None: continue
        key = int(img_dist / tolerance)
        candidate_pairs = catalog_hash.get(key, [])
        for hip1, hip2 in candidate_pairs:
            votes[(s1, hip1)] += 1
            votes[(s1, hip2)] += 1
            votes[(s2, hip1)] += 1
            votes[(s2, hip2)] += 1
    return votes


def lost_in_space():
    # --- SETUP (Same as before) ---
    print("Loading catalog data...")
    with open('catalog_hash.pkl', 'rb') as f: catalog_hash = pickle.load(f)
    all_catalog_angular_distances = load_catalog_angular_distances()
    print("Data loaded.")

    # --- IMAGE PROCESSING (Same as before) ---
    star_coords = ip.find_stars_with_advanced_filters(IMAGE_FILE, NUM_STARS)

    img_unit_vectors = star_coords_to_unit_vector(star_coords, (CENTER_X, CENTER_Y), FOCAL_LENGTH_X, FOCAL_LENGTH_Y)
    img_ang_dists = get_angular_distances(star_coords, (CENTER_X, CENTER_Y), FOCAL_LENGTH_X, FOCAL_LENGTH_Y)

    # --- STAGE 1: Generate Raw Votes ---
    TOLERANCE_ACQUISITION = 1.5
    raw_votes = generate_raw_votes(img_ang_dists, catalog_hash, TOLERANCE_ACQUISITION)
    if not raw_votes:
        print("!!! FAILED: No votes generated."); return None
    
    # --- STAGE 2: Build Hypothesis List from Votes ---
    # This filters out the low-vote noise from the "vote hogs".
    hypotheses = build_hypotheses_from_votes(raw_votes)
    if not hypotheses:
        print("!!! FAILED: No hypotheses met the minimum vote threshold."); return None
    print(f"Built hypotheses for {len(hypotheses)} stars from raw votes.")

    sorted_hypothesises = dict(sorted(hypotheses.items(), key=lambda item: len(item[1])))
    image_stars = list(hypotheses.keys())
    solutions = []

    print("Starting DFS to find a consistent solution (max 10 seconds)...")
    ip.display_star_detections(IMAGE_FILE, star_coords)
    # Capture the start time right before the first call
    dfs_start_time = time.time()
    DFS(
        {}, # Start with empty assignment
        image_stars,
        sorted_hypothesises,
        img_ang_dists,
        all_catalog_angular_distances,
        TOLERANCE_ACQUISITION,
        solutions,
        dfs_start_time, # Pass the start time
        max_time=10   # Set the max duration
    )

    print(f"DFS finished in {time.time() - dfs_start_time:.2f} seconds.")
    # --- STAGE 4: Score Solutions and Calculate Attitude ---
    if not solutions:
        print("!!! FAILED: DFS could not find any geometrically consistent solution.")
        return None

    # Use your existing scoring and sorting logic
    scored_solutions = load_solution_scoring(solutions, img_ang_dists, all_catalog_angular_distances)
    # Sort by number of stars in solution (descending), then score (ascending)
    sorted_arr = sorted(scored_solutions, key=lambda x: (-len(x[0]), x[1]))
    best_match = sorted_arr[0]
    final_solution = best_match[0]
    
    print(f"SUCCESS: DFS found {len(solutions)} solution(s). Best one has {len(final_solution)} stars:", final_solution)
    
    # --- STAGE 5: Calculate Attitude (Same as before) ---
    cat_unit_vectors = load_catalog_unit_vectors("src/test/star_distances_sorted.db")
    mapped_image_vectors = []
    for key in final_solution.keys():
        mapped_image_vectors.append(img_unit_vectors[key])
    img_matrix = image_vector_matrix(mapped_image_vectors)
    # This wrapper is no longer needed if you simplify catalog_vector_matrix
    cat_matrix = catalog_vector_matrix({'solution': final_solution}, cat_unit_vectors) 

    quaternion = compute_attitude_quaternion(img_matrix, cat_matrix)
    print("Final Quaternion Calculated.")
    ip.display_star_detections(IMAGE_FILE, star_coords)
    return quaternion

if __name__ == "__main__":
    start_time = time.time()
    q = lost_in_space()
    end_time = time.time()
    print(f"{end_time - start_time}")
    print(f"{q}")
