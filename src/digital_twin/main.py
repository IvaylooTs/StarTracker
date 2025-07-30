import cv2
import numpy as np
import sqlite3
import math
from itertools import combinations
from itertools import product

IMAGE_HEIGHT = 1964
IMAGE_WIDTH = 3024
ASPECT_RATIO = IMAGE_HEIGHT / IMAGE_WIDTH
FOV_Y = 20
FOV_X = math.degrees(2 * math.atan(math.tan(math.radians(FOV_Y / 2)) / ASPECT_RATIO))
CENTER_X = IMAGE_WIDTH/2
CENTER_Y = IMAGE_HEIGHT/2
FOCAL_LENGTH_X = (IMAGE_WIDTH / 2) / math.tan(math.radians(FOV_X / 2))
FOCAL_LENGTH_Y = (IMAGE_HEIGHT / 2) / math.tan(math.radians(FOV_Y / 2))
TOLERANCE = 3
IMAGE_FILE = "./test_images/testing23.png"
NUM_STARS = 10
EPSILON = 1e-6

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

    print(f"Found {len(keypoints)} valid stars. Returning the {len(brightest_stars_coords)} brightest.")
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
    print(f"Actual size: {img.shape[1]} x {img.shape[0]}")

    star_coords = find_brightest_stars(IMAGE_FILE, NUM_STARS)

    print(f"☆ Star pixel coordinates:\n")
    if star_coords:
        print(f"\nCoordinates of the {len(star_coords)} brightest stars:")
        for i, (x, y) in enumerate(star_coords):
            print(f"  Star {i+1}: (x={x:.4f}, y={y:.4f})")
        
    print(f"\n☆ Image pair angular distance:\n")
    ang_dists = get_angular_distances(star_coords, (CENTER_X, CENTER_Y), FOCAL_LENGTH_X, FOCAL_LENGTH_Y)
    for (s1,s2), ang_dist in ang_dists.items():
        print(f"{s1}->{s2}: {ang_dist}")
    
    hypothesises = load_hypothesies(NUM_STARS, ang_dists, TOLERANCE)
    
    
    print(f"☆ Candidates for each star:\n")
    for star, elements in hypothesises.items():
        print(f"image star: {star} -> hips: {elements}")
        print("\n")
        
    a = generate_geometric_hashes(star_coords)
    for key, val in a.items():
        print(f"{key} -> {val}")
    
    # assignment = {}
    # image_stars = []
    # for i in range(0, NUM_STARS):
    #     image_stars.append(i)
    # catalog_dists = catalog_angular_distances(ang_dists)
    # result = DFS(assignment, image_stars, hypothesises, ang_dists, catalog_dists, TOLERANCE)
    # if result:
    #     print("\n☆ Final assignment (image star → HIP):")
    #     for image_star, hip in result.items():
    #         print(f"  Image Star {image_star} → HIP {hip}")
    # else:
    #     print("☆ No valid assignment found. Try increasing tolerance or reducing number of stars.")
        
    
    # display_star_detections(IMAGE_FILE, star_coords)