import cv2
import numpy as np
import sqlite3
import math
from itertools import combinations
from itertools import product

# ==============================================================================
# --- CAMERA & SENSOR CONFIGURATION ---
# ==============================================================================
IMAGE_HEIGHT = 1964
IMAGE_WIDTH = 3024
ASPECT_RATIO = IMAGE_HEIGHT / IMAGE_WIDTH
FOV_Y = 20
FOV_X = math.degrees(2 * math.atan(math.tan(math.radians(FOV_Y / 2)) / ASPECT_RATIO))
CENTER_X = IMAGE_WIDTH / 2
CENTER_Y = IMAGE_HEIGHT / 2
FOCAL_LENGTH_X = (IMAGE_WIDTH / 2) / math.tan(math.radians(FOV_X / 2))
FOCAL_LENGTH_Y = (IMAGE_HEIGHT / 2) / math.tan(math.radians(FOV_Y / 2))
TOLERANCE = 3
IMAGE_FILE = "./test_images/testing23.png"
NUM_STARS = 10
EPSILON = 1e-6

# ==============================================================================
# --- NEW: STAR DETECTION PARAMETERS ---
# ==============================================================================
BINARY_THRESHOLD = 87      # Keep this high to isolate bright objects
MIN_STAR_AREA = 10
MAX_STAR_AREA = 400        # Filter out very large objects
MIN_CIRCULARITY = 0.80     # How close to a perfect circle (1.0). Good for spiky noise.
MAX_ECCENTRICITY = 0.5     # 0 = circle, closer to 1 = elongated.
MIN_SOLIDITY = 0.98        # How "solid" the shape is. 1.0 has no holes/dents. Rejects notches.
MIN_PEAK_RATIO = 0.9       # Ratio of core brightness to surrounding area brightness.


# ==============================================================================
# --- REPLACED IMAGE FILTER FUNCTION ---
# ==============================================================================
def find_stars_with_advanced_filters(
    image_path: str,
    n_stars: int,
    binary_threshold: int,
    min_area: int,
    max_area: int,
    min_circularity: float,
    max_eccentricity: float,
    min_solidity: float,
    min_peak_ratio: float
    ):
    """
    Identifies stars using a multi-stage filtering pipeline.

    Returns:
        A NumPy array of (x, y) coordinates for the brightest N stars.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # --- Step 1: Find Initial Candidates ---
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary_image = cv2.threshold(blurred, binary_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Initial detection: Found {len(contours)} potential objects.")

    candidates_after_shape_filter = []

    # --- Step 2: The Shape Gauntlet ---
    for contour in contours:
        # Filter 2a: Area
        area = cv2.contourArea(contour)
        if not (min_area < area < max_area):
            continue

        # Filter 2b: Circularity
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0: continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity < min_circularity:
            continue

        # Filter 2c: Eccentricity (Ellipse Fitting)
        if len(contour) >= 5: # cv2.fitEllipse requires at least 5 points
            ellipse = cv2.fitEllipse(contour)
            (x, y), (MA, ma), angle = ellipse
            if MA == 0: continue # Avoid division by zero
            # Eccentricity formula: e = sqrt(1 - (b/a)^2) where a is major axis, b is minor
            eccentricity = np.sqrt(1 - (min(MA, ma)**2 / max(MA, ma)**2))
            if eccentricity > max_eccentricity:
                continue
        else:
            # If the contour is too small to fit an ellipse, reject it.
            continue

        # Filter 2d: Solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        solidity = float(area) / hull_area
        if solidity < min_solidity:
            continue

        candidates_after_shape_filter.append(contour)

    print(f"After shape gauntlet (area, circularity, eccentricity, solidity): {len(candidates_after_shape_filter)} candidates remain.")

    valid_stars = []

    # --- Step 3: Peak Brightness Filter ---
    for contour in candidates_after_shape_filter:
        M = cv2.moments(contour)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        inner_radius = 2
        outer_radius = 5

        mask = np.zeros(image.shape, dtype=np.uint8)
        outer_mask = cv2.circle(mask.copy(), (cx, cy), outer_radius, 255, -1)
        inner_mask = cv2.circle(mask.copy(), (cx, cy), inner_radius, 255, -1)
        donut_mask = outer_mask & ~inner_mask

        try:
            avg_inner_brightness = np.mean(image[inner_mask == 255])
            avg_outer_brightness = np.mean(image[donut_mask == 255])

            if avg_outer_brightness == 0:
                if avg_inner_brightness > binary_threshold:
                    valid_stars.append(((cx, cy), cv2.contourArea(contour)))
                continue

            brightness_ratio = avg_inner_brightness / avg_outer_brightness

            if brightness_ratio > min_peak_ratio:
                valid_stars.append(((cx, cy), cv2.contourArea(contour)))

        except (ValueError, ZeroDivisionError):
            continue

    print(f"After peak brightness filtering: {len(valid_stars)} valid stars remain.")

    # --- Step 4: Sort by Brightness/Size and Select Top N ---
    valid_stars.sort(key=lambda s: s[1], reverse=True)
    brightest_stars = valid_stars[:n_stars]
    centroids = np.array([star[0] for star in brightest_stars]) if brightest_stars else np.array([])

    # Return only the centroids to match the requirements of the main script
    return centroids

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

def star_coords_to_unit_vector(star_coords, center_coords, f_x, f_y):
    cx, cy = center_coords
    unit_vectors = []
    for (x, y) in star_coords:
        dx = x - cx
        dy = y - cy
        direction_vector = np.array([dx, dy, f_x])
        normalized_vector = direction_vector / np.linalg.norm(direction_vector)
        unit_vectors.append(normalized_vector)
    return unit_vectors

def angular_dist_helper(unit_vectors):
    num_of_vectors = len(unit_vectors)
    angular_distances = {}
    for i in range(0, num_of_vectors):
        for j in range(i + 1, num_of_vectors):
            vector1 = unit_vectors[i]
            vector2 = unit_vectors[j]
            
            dot_product = np.dot(vector1, vector2)
            dot__product_clamped = np.clip(dot_product, -1.0, 1.0)
            
            angular_dist_rad = np.arccos(dot__product_clamped)
            angular_dist_deg = np.degrees(angular_dist_rad)
            angular_distances[(i, j)] = angular_dist_deg
    return angular_distances

def get_angular_distances(star_coords, center_coords, f_x, f_y):
    unit_vectors = star_coords_to_unit_vector(star_coords, center_coords, f_x, f_y)
    return angular_dist_helper(unit_vectors)

def load_catalog_angular_distances(min_distance=0.1, max_distance=0.5, db_path='star_distances_sorted.db', table_name='AngularDistances'):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        query = f"""
            SELECT hip1, hip2, angular_distance
            FROM {table_name}
            WHERE angular_distance BETWEEN ? AND ?
        """
        cursor.execute(query, (min_distance, max_distance))
        rows = cursor.fetchall()

        return { (row[0], row[1]): row[2] for row in rows }

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {}

    finally:
        if conn:
            conn.close()

def catalog_angular_distances(angular_distances):
    catalog_ang_dists = {}
    for (s1, s2), ang_dist in angular_distances.items():
        min_ang_dist = ang_dist - TOLERANCE
        max_ang_dist = ang_dist + TOLERANCE
        cur_cat_ang_dist = load_catalog_angular_distances(min_ang_dist, max_ang_dist)
        catalog_ang_dists[(s1,s2)] = cur_cat_ang_dist
    return catalog_ang_dists

def within_bounds(ang_dist, tolerance):
    min_max = []
    min_max.append(ang_dist - tolerance)
    min_max.append(ang_dist + tolerance)
    return min_max

def load_hypothesies(num_stars, angular_distances, tolerance):
    hypothesises_dict = {i: set() for i in range(num_stars)}

    for i in range(num_stars):
        for j in range(i + 1, num_stars):
            pair = (i, j)
            cur_ang_dist = angular_distances.get(pair)
            if cur_ang_dist is None:
                continue

            bounded_ang_dist = within_bounds(cur_ang_dist, tolerance)
            cur_dict = load_catalog_angular_distances(bounded_ang_dist[0], bounded_ang_dist[1])

            for (h1, h2), _ in cur_dict.items():
                hypothesises_dict[i].update([h1, h2])
                hypothesises_dict[j].update([h1, h2])

    return hypothesises_dict

def triplet_properties(triplet, star_coords):
    a, b, c = triplet
    d_ab = math.dist(star_coords[a], star_coords[b])
    d_ac = math.dist(star_coords[a], star_coords[c])
    d_bc = math.dist(star_coords[b], star_coords[c])
    
    p = (d_ab + d_ac + d_bc) / 2
    area = math.sqrt(p*(p - d_ab)*(p - d_ac)*(p - d_bc))
    
    elongation = max(d_ab, d_ac, d_bc) / max(min(d_ab, d_ac, d_bc), EPSILON)

    return d_ab, d_ac, d_bc, area, elongation

def is_degenerate_triplet(triplet, min_area = EPSILON, max_elongation = 10):
    _, _, _, area, elongation = triplet
    
    if area < min_area:
        return True
    
    if elongation > max_elongation:
        return True
    
    return False

def hash_key(triplet):
    a, b, c, _, _ = triplet
    sides = sorted([a, b, c])
    norm = sides[-1]
    normalized_sides = [side / norm for side in sides]
    rounded_sides = tuple(round(x, 5) for x in normalized_sides)
    return rounded_sides

def generate_geometric_hashes(image_coords):
    triplets = list(combinations(range(NUM_STARS), 3))
    filtered_triplets = {}
    
    for triplet in triplets:
        triplet_props = triplet_properties(triplet, image_coords)
        if not is_degenerate_triplet(triplet_props):
            key = hash_key(triplet_props)
            if key not in filtered_triplets:
                filtered_triplets[key] = set()
            filtered_triplets[key].update(triplet)
            
    return filtered_triplets

def DFS(assignment, image_stars, candidate_hips, image_angular_distances, catalog_angular_distances, tolerance):
    if len(assignment) == len(image_stars):
        return assignment 

    current_star = next(s for s in image_stars if s not in assignment)

    for hip_candidate in candidate_hips.get(current_star, []):
        if hip_candidate in assignment.values():
            continue 

        assignment[current_star] = hip_candidate
        
        if is_consistent(assignment, image_angular_distances, catalog_angular_distances, tolerance):
            result = DFS(assignment, image_stars, candidate_hips, image_angular_distances, catalog_angular_distances, tolerance)
            if result is not None:
                return result

        del assignment[current_star]

    return None

def is_consistent(assignment, image_distances, catalog_distances, tolerance):
    for (i1, i2), img_angle in image_distances.items():
        if i1 in assignment and i2 in assignment:
            hip1 = assignment[i1]
            hip2 = assignment[i2]

            cat_subdict = catalog_distances.get((i1, i2), {})
            catalog_angle = cat_subdict.get((hip1, hip2)) or cat_subdict.get((hip2, hip1))

            if catalog_angle is None:
                return False

            if abs(catalog_angle - img_angle) > tolerance:
                return False
    return True

if __name__ == "__main__":
    img = cv2.imread(IMAGE_FILE)
    if img is None:
        print(f"FATAL ERROR: Could not load image from '{IMAGE_FILE}'")
        exit() # Exit immediately if the image can't be found

    print(f"Actual size: {img.shape[1]} x {img.shape[0]}")

    # --- MODIFIED: Calling the new, advanced star detection function ---
    star_coords = find_stars_with_advanced_filters(
        IMAGE_FILE,
        NUM_STARS,
        BINARY_THRESHOLD,
        MIN_STAR_AREA,
        MAX_STAR_AREA,
        MIN_CIRCULARITY,
        MAX_ECCENTRICITY,
        MIN_SOLIDITY,
        MIN_PEAK_RATIO
    )

    print(f"\n☆ Star pixel coordinates:")
    if len(star_coords) > 0:
        print(f"Coordinates of the {len(star_coords)} brightest stars:")
        for i, (x, y) in enumerate(star_coords):
            print(f"  Star {i}: (x={x:.4f}, y={y:.4f})")

        # --- Continue with geometric analysis and database lookups ---
        print(f"\n☆ Image pair angular distance:")
        ang_dists = get_angular_distances(star_coords, (CENTER_X, CENTER_Y), FOCAL_LENGTH_X, FOCAL_LENGTH_Y)
        for (s1,s2), ang_dist in ang_dists.items():
            print(f"  {s1} -> {s2}: {ang_dist:.4f} degrees")

        # --- Load candidate HIPs from the database based on angular distances ---
        hypothesises = load_hypothesies(len(star_coords), ang_dists, TOLERANCE)
        print(f"\n☆ Candidates for each star:")
        for star, elements in hypothesises.items():
            # Truncate long lists of candidates for better readability
            if len(elements) > 25:
                 print(f"  Image star {star} -> {len(elements)} HIP candidates: {list(elements)[:25]}...")
            else:
                 print(f"  Image star {star} -> HIPs: {elements}")


        # --- Generate geometric hashes from star triplets ---
        print(f"\n☆ Generated Geometric Hashes:")
        geometric_hashes = generate_geometric_hashes(star_coords)
        for key, val in geometric_hashes.items():
            print(f"  Hash {key} -> stars {val}")


        # --- FINAL MATCHING: This section performs the DFS to find a consistent assignment. ---
        # --- It is commented out by default as it can be slow. Uncomment to run. ---
        #
        # print("\n--- Attempting to find a consistent star assignment... ---")
        # assignment = {}
        # image_stars = list(range(len(star_coords)))
        # catalog_dists = catalog_angular_distances(ang_dists) # This query can be slow
        #
        # result = DFS(assignment, image_stars, hypothesises, ang_dists, catalog_dists, TOLERANCE)
        #
        # if result:
        #     print("\n☆ Final assignment (image star → HIP):")
        #     for image_star, hip in sorted(result.items()):
        #         print(f"  Image Star {image_star} → HIP {hip}")
        # else:
        #     print("\n☆ No valid assignment found. Try increasing tolerance or checking star detection parameters.")


        # --- VISUALIZATION: This section displays the identified stars on the image. ---
        # --- Uncomment to show the final image and save it to a file. ---
        #
        # print("\n--- Displaying final detections... ---")
        # display_star_detections(IMAGE_FILE, star_coords)

    else:
        # This runs if the star detection function returns an empty list
        print("\nNo stars were detected that met all the filter criteria. Halting script.")