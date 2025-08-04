import cv2
import numpy as np
import sqlite3
import math
from itertools import combinations
from itertools import product
from collections import defaultdict

IMAGE_HEIGHT = 1964
IMAGE_WIDTH = 3024
ASPECT_RATIO = IMAGE_HEIGHT / IMAGE_WIDTH
FOV_Y = 53
FOV_X = math.degrees(2 * math.atan(math.tan(math.radians(FOV_Y / 2)) / ASPECT_RATIO))
CENTER_X = IMAGE_WIDTH/2
CENTER_Y = IMAGE_HEIGHT/2
FOCAL_LENGTH_X = (IMAGE_WIDTH / 2) / math.tan(math.radians(FOV_X / 2))
FOCAL_LENGTH_Y = (IMAGE_HEIGHT / 2) / math.tan(math.radians(FOV_Y / 2))
TOLERANCE = 2
IMAGE_FILE = "./test_images/testing34.png"
NUM_STARS = 10
EPSILON = 1e-6
MIN_MATCHES = 5

#tape fix

def find_brightest_stars(image_path: str, num_stars_to_find: int):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at '{image_path}'")
        return []

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 4
    params.maxArea = 500

    params.filterByCircularity = True
    params.minCircularity = 0.5

    params.filterByConvexity = True
    params.minConvexity = 0.8

    params.filterByInertia = True
    params.minInertiaRatio = 0.8

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(255 - img_gray)

    if not keypoints:
        print("Warning: No blobs passed the detector's filters.")
        return []

    keypoints = sorted(keypoints, key=lambda k: k.size, reverse=True)
    brightest_stars_coords = [kp.pt for kp in keypoints[:num_stars_to_find]]

    # print(f"Found {len(keypoints)} valid stars. Returning the {len(brightest_stars_coords)} brightest.")
    return brightest_stars_coords

def display_star_detections(image_path: str, star_coords: list, output_filename: str = 'stars_identified.png'):
    img_color = cv2.imread(image_path)
    if img_color is None:
        print(f"Error: Cannot load image for display at '{image_path}'")
        return

    cross_color = (0, 0, 255)
    circle_color = (255, 204, 229)
    text_color = (255, 204, 229)
    cross_thickness = 1
    circle_radius = 30
    circle_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2

    for idx, (x, y) in enumerate(star_coords):
        x_int, y_int = int(round(x)), int(round(y))

        cv2.circle(img_color, (x_int, y_int), circle_radius, circle_color, circle_thickness)
        cv2.drawMarker(img_color, (x_int, y_int), cross_color, markerType=cv2.MARKER_CROSS, thickness=cross_thickness)

        text_position = (x_int + 10, y_int - 10)
        cv2.putText(img_color, str(idx), text_position, font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    cv2.imwrite(output_filename, img_color)
    print(f"\nVisual result saved to '{output_filename}'.")

    cv2.imshow('Identified Stars', img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#End of tape fix

# Unit vector function -> finds star unit vectros based on star pixel coordinates using pinhole projection
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
    for (x, y) in star_coords:
        # Pick arbitrary depth for each 3D point (we don't know how far away they really are, we just want their direction)
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
    angular_dists = {(i, j): angular_dist_matrix[i, j]
                     for i, j in combinations(range(len(unit_vectors)), 2)}
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
def load_catalog_angular_distances(db_path='star_distances_sorted.db', table_name='AngularDistances'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = f"SELECT hip1, hip2, angular_distance FROM {table_name}"
    cursor.execute(query)
    rows = cursor.fetchall()

    conn.close()

    return { (row[0], row[1]): row[2] for row in rows }

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

# Function that returns a dict {(image star index 1, image star index 2): dict2}
# dict2 - {(HIP ID 1, HIP ID 2): angular distnace}
# Inputs:
# - angular_distances: dict where {(image star index 1, image star index 2): image angular distance}
# - all_cat_ang_dists: dict where {(HIP ID 1, HIP ID 2): catalog angular distance}
# - tolerance: angular distance tolerance
def catalog_angular_distances(angular_distances, all_cat_ang_dists, tolerance):
    catalog_ang_dists = {}
    for (s1, s2), ang_dist in angular_distances.items():
        bounds = get_bounds(ang_dist, tolerance)
        cur_cat_ang_dist = filter_catalog_angular_distances(all_cat_ang_dists, bounds)
        catalog_ang_dists[(s1,s2)] = cur_cat_ang_dist
    return catalog_ang_dists

# Function that loads candidate catalog HIP IDs for each image star
# Inputs:
# - num_stars: number of stars from the image
# - angular_distances: dict where {(image star index 1, image star index 2): image angular distance}
# - all_cat_ang_dists: dict where {(HIP ID 1, HIP ID 2): catalog angular distance}
# - tolerance: angular distance tolerance
# Outputs:
# - hypotheses_dict: {image star index: [HIP ID 1, ... HIP ID N]}
def load_hypotheses(angular_distances, all_cat_ang_dists, tolerance):
    hypotheses_dict = defaultdict(set)

    for (i, j), cur_ang_dist in angular_distances.items():
        if cur_ang_dist is None:
            continue

        bounds = get_bounds(cur_ang_dist, tolerance)
        cur_dict = filter_catalog_angular_distances(all_cat_ang_dists, bounds)

        for h1, h2 in cur_dict.keys():
            hypotheses_dict[i].update([h1, h2])
            hypotheses_dict[j].update([h1, h2])

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
def DFS(assignment, image_stars, candidate_hips, image_angular_distances, catalog_angular_distances, tolerance, solutions):
    
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
        if is_consistent(assignment, image_angular_distances, catalog_angular_distances, tolerance, current_star):
            DFS(assignment, image_stars, candidate_hips, image_angular_distances, catalog_angular_distances, tolerance, solutions)
        
        del assignment[current_star]

# Bool function that checks if the current assignment is consistent with the angular distances from our image within a certain tolerance
# Inputs:
# - assignment: current assignment
# - image_angular_distances: dict where {(image star index 1, image star index 2): image angular distance}
# - catalog_angular_distances: dict where {(HIP ID 1, HIP ID 2): catalog angular distance}
# - tolerance: angular distance tolerance
# - new_star: index of new image star being added to the assignment
def is_consistent(assignment, image_angular_distances, catalog_angular_distances, tolerance, new_star):
    
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

            catalog_angle = catalog_angular_distances.get((hip1, hip2)) or catalog_angular_distances.get((hip2, hip1))

            if catalog_angle is None:
                return False

            if abs(catalog_angle - img_angle) > tolerance:
                return False
    return True

def score_solution(solution, image_angular_distances, catalog_angular_distances, tolerance=TOLERANCE):
    total_diff = 0
    count = 0

    for (s1, s2), img_ang_dist in image_angular_distances.items():

        if s1 in solution and s2 in solution:
            hip1, hip2 = solution[s1], solution[s2]
            
            subdict = catalog_angular_distances.get((s1, s2))
            key = (hip1, hip2) if (hip1, hip2) in subdict else (hip2, hip1)
            cat_ang_dist = subdict.get(key)
    
            if cat_ang_dist is None:
                continue

            diff = abs(cat_ang_dist - img_ang_dist)
            diff = min(diff, 360 - diff)
            total_diff += diff
            count += 1
    
    if count == 0:
        return float('inf')

    coverage = count / len(image_angular_distances)
    score = (total_diff / count) * (1 / coverage)
    return score

def load_solution_scoring(solutions, image_angular_distances, catalog_angular_distances):
    scored_solutions = []
    for sol in solutions:
        score = score_solution(sol, image_angular_distances, catalog_angular_distances)
        scored_solutions.append(tuple((sol, score)))
    return scored_solutions

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

def build_B_matrix(image_vectors, catalog_vectors, weights=None):
    
    assert image_vectors.shape == catalog_vectors.shape, "Shape mismatch between image and catalog vectors"
    
    N = image_vectors.shape[0]
    if weights is None:
        weights = np.ones(N)
    
    B = np.zeros((3, 3))
    for i in range(N):
        B += weights[i] * np.outer(image_vectors[i], catalog_vectors[i])
    
    return B

def build_K_matrix(B):
    S = B + B.T
    sigma = np.trace(B)
    Z = np.array([
        B[1, 2] - B[2, 1],
        B[2, 0] - B[0, 2],
        B[0, 1] - B[1, 0]
    ])

    K = np.zeros((4, 4))
    K[0, 0] = sigma
    K[0, 1:] = Z
    K[1:, 0] = Z
    K[1:, 1:] = S - sigma * np.eye(3)
    return K

def find_optimal_quaternion(K):
    eigenvalues, eigenvectors = np.linalg.eigh(K)
    max_index = np.argmax(eigenvalues)
    optimal_quaternion = eigenvectors[:, max_index]
    optimal_quaternion /= np.linalg.norm(optimal_quaternion)
    return optimal_quaternion

def compute_attitude_quaternion(image_vectors, catalog_vectors, weights=None):
    B = build_B_matrix(image_vectors, catalog_vectors, weights)
    K = build_K_matrix(B)
    q = find_optimal_quaternion(K)
    
    return q
   
def lost_in_space():
    
    star_coords = find_brightest_stars(IMAGE_FILE, NUM_STARS)
    img_unit_vectors = star_coords_to_unit_vector(star_coords, (CENTER_X, CENTER_Y), FOCAL_LENGTH_X, FOCAL_LENGTH_Y)
    ang_dists = get_angular_distances(star_coords, (CENTER_X, CENTER_Y), FOCAL_LENGTH_X, FOCAL_LENGTH_Y)
    for (i, j), and_dist in ang_dists.items():
        print(f"{i}->{j}: {ang_dists}")

    all_cat_ang_dists = load_catalog_angular_distances()
    hypotheses = load_hypotheses(ang_dists, all_cat_ang_dists, TOLERANCE)
    sorted_hypothesises = dict(sorted(hypotheses.items(), key=lambda item: len(item[1])))
        
    assignment = {}
    image_stars = []
    
    for i in range(0, NUM_STARS):
        image_stars.append(i)
        
    catalog_dists = catalog_angular_distances(ang_dists, all_cat_ang_dists, TOLERANCE)
    
    solutions = []
    DFS(assignment, image_stars, sorted_hypothesises, ang_dists, all_cat_ang_dists, TOLERANCE, solutions)
    
    scored_solutions = load_solution_scoring(solutions, ang_dists, catalog_dists)
    
    sorted_arr = sorted(scored_solutions, key=lambda x: x[1])
    best_match = sorted_arr[0]
    
    print(f"Best match: ")
    print(f"{best_match[0]}")
    
    cat_unit_vectors = load_catalog_unit_vectors("star_distances_sorted.db")
    
    new_img_vectors = []
    for i in range(0, len(best_match[0])):
        new_img_vectors.append(img_unit_vectors[i])
    img_matrix = image_vector_matrix(new_img_vectors)
    cat_matrix = catalog_vector_matrix(best_match, cat_unit_vectors)
    
    quaternion = compute_attitude_quaternion(img_matrix, cat_matrix)
    display_star_detections(IMAGE_FILE, star_coords)
    return quaternion
     
if __name__ == "__main__":
    q = lost_in_space()
    print(f"{q}")
