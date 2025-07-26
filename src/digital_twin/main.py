import cv2
import numpy as np
import sqlite3
import math

FOV = 66
IMAGE_WIDTH = 1964
IMAGE_HEIGHT = 3024
CENTER_X = IMAGE_WIDTH/2
CENTER_Y = IMAGE_HEIGHT/2
FOCAL_LENGTH = (IMAGE_WIDTH / 2) / math.tan(math.radians(FOV / 2))
TOLERANCE = 0.1

def star_coords_to_unit_vector(star_coords, center_coords, focal_length):
    cx, cy = center_coords
    unit_vectors = []
    for (x, y) in star_coords:
        centered_x = (x - cx) / focal_length
        centered_y = (y - cy) / focal_length
        direction_vector = np.array([centered_x, centered_y, 1])
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

def get_angular_distances(star_coords, center_coords, focal_length):
    unit_vectors = star_coords_to_unit_vector(star_coords, center_coords, focal_length)
    return angular_dist_helper(unit_vectors)

# def distance1(x1, y1, x2, y2):
#     return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# def angular_distances2(arg_star_coords):
#     angular_distances = {}
#     for i in range(len(arg_star_coords)):
#         x1, y1 = arg_star_coords[i]
#         for j in range(i + 1, len(arg_star_coords)):
#             x2, y2 = arg_star_coords[j]
#             distance = distance1(x1, y1, x2, y2)
#             ang_dist = distance / IMAGE_WIDTH * FOV
#             angular_distances[(i, j)] = ang_dist
#     return angular_distances

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

def within_bounds(ang_dist, tolerance):
    min_max = []
    min_max.append(ang_dist - tolerance)
    min_max.append(ang_dist + tolerance)
    return min_max

def load_hypothesies(num_stars, angular_distances, tolerance):
    hypothesises_dict = {}

    for i in range(0, num_stars):
        for j in range(i + 1, num_stars):
            pair = (i, j)
            cur_ang_dist = angular_distances.get(pair)
            if cur_ang_dist is None:
                continue
            bounded_ang_dist = within_bounds(cur_ang_dist, tolerance)
            cur_dict = load_catalog_angular_distances(bounded_ang_dist[0], bounded_ang_dist[1])  

        for (h1, h2), dist in cur_dict.items():
            hypothesises_dict.setdefault(i, set()).update([h1, h2])
            hypothesises_dict.setdefault(j, set()).update([h1, h2])
    
    return hypothesises_dict

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
    cross_thickness = 1
    circle_radius = 30
    circle_thickness = 2

    for (x, y) in star_coords:
        x_int, y_int = int(round(x)), int(round(y))

        cv2.circle(img_color, (x_int, y_int), circle_radius, circle_color, circle_thickness)
        cv2.drawMarker(img_color, (x_int, y_int), cross_color, markerType=cv2.MARKER_CROSS, thickness=cross_thickness)

    cv2.imwrite(output_filename, img_color)
    print(f"\nVisual result saved to '{output_filename}'.")

    cv2.imshow('Identified Stars', img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#End of tape fix

if __name__ == "__main__":
    IMAGE_FILE = "./test_images/testing3.png"
    NUM_STARS = 10

    star_coords = find_brightest_stars(IMAGE_FILE, NUM_STARS)

    if star_coords:
        print(f"\nCoordinates of the {len(star_coords)} brightest stars:")
        for i, (x, y) in enumerate(star_coords):
            print(f"  Star {i+1}: (x={x:.4f}, y={y:.4f})")
        
    angular_distances = get_angular_distances(star_coords, (CENTER_X, CENTER_Y), FOCAL_LENGTH)
    catalog_ang_dists = []
    for (s1, s2), ang_dist in angular_distances.items():
        min_ang_dist = ang_dist - TOLERANCE
        max_ang_dist = ang_dist + TOLERANCE
        cur_cat_ang_dist = load_catalog_angular_distances(min_ang_dist, max_ang_dist)
        catalog_ang_dists.append(cur_cat_ang_dist)
    
    for element in catalog_ang_dists:
        for (hip1, hip2), ang_dist in element.items():
            print(f"{hip1}->{hip2}: {ang_dist}")
        print("\n")
    
    hypothesises = load_hypothesies(NUM_STARS, angular_distances, TOLERANCE)
    
    for star, elements in hypothesises.items():
        print(f"{star} -> {elements}")
        print("\n")
