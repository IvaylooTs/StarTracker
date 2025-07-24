import cv2
import numpy as np
import math

FOV = 66
IMAGE_WIDTH = 1964
IMAGE_HEIGHT = 3024
CENTER_X = IMAGE_WIDTH/2
CENTER_Y = IMAGE_HEIGHT/2
FOCAL_LENGTH = (IMAGE_WIDTH / 2) / math.tan(math.radians(FOV / 2))

def star_coords_to_unit_vector(arg_star_coords, arg_center_coords, arg_focal_length):
    cx, cy = arg_center_coords
    unit_vectors = []
    for (x, y) in arg_star_coords:
        centered_x = (x - cx) / arg_focal_length
        centered_y = (y - cy) / arg_focal_length
        direction_vector = np.array([centered_x, centered_y, 1])
        normalized_vector = direction_vector / np.linalg.norm(direction_vector)
        unit_vectors.append(normalized_vector)
    return unit_vectors

def angular_distances(unit_vectors):
    num_of_vectors = len(unit_vectors)
    angular_distances = {}
    for i in range(0, num_of_vectors):
        for j in range(i + 1, num_of_vectors):
            vector1 = unit_vectors[i]
            vector2 = unit_vectors[j]
            
            dot_product = np.dot(vector1, vector2)
            dot__product_clamped = np.clip(dot_product, -1.0, 1.0)
            
            angular_distance = np.arccos(dot__product_clamped)
            angular_deg = np.degrees(angular_distance)
            angular_distances[(i, j)] = angular_deg
    return angular_distances

def distance1(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def angular_distances2(arg_star_coords):
    angular_distances = {}
    for i in range(len(arg_star_coords)):
        x1, y1 = arg_star_coords[i]
        for j in range(i + 1, len(arg_star_coords)):
            x2, y2 = arg_star_coords[j]
            distance = distance1(x1, y1, x2, y2)
            ang_dist = distance / IMAGE_WIDTH * FOV
            angular_distances[(i, j)] = ang_dist
    return angular_distances
        
        
def geometrically_consistent(image_to_catalog_assignment, image_pairs_to_ang_dist, catalog_ang_dist, tolerance):
    for (star1, star2), ang_dist in image_pairs_to_ang_dist:
        if star1 in image_to_catalog_assignment and star2 in image_to_catalog_assignment:
            cat_star1 = image_to_catalog_assignment[star1]
            cat_star2 = image_to_catalog_assignment[star2]
            cat_ang_dist = catalog_ang_dist.get((cat_star1, cat_star2)) or catalog_ang_dist.get((cat_star2, cat_star1))
            if cat_ang_dist is None or abs(cat_ang_dist - ang_dist) > tolerance:
                return False
    return True



def get_possible_catalogs(current, graph_hypotheses):
    possible_catalogs = set()
    for (i1, i2), cat_pairs in graph_hypotheses.items():
        if current == i1:
            possible_catalogs.update(c[0] for c in cat_pairs)
        elif current == i2:
            possible_catalogs.update(c[1] for c in cat_pairs)
    return possible_catalogs

def DFS(image_to_catalog_assignment, image_pairs_to_ang_dist, catalog_ang_dist, tolerance, graph_hypotheses, image_stars):
    if len(image_to_catalog_assignment) == len(image_stars):
        return image_pairs_to_ang_dist
    
    current = next(s for s in image_stars if s not in image_to_catalog_assignment)
    possible_catalogs = get_possible_catalogs(current, graph_hypotheses)
    for catalog_candidate in possible_catalogs:
        for catalog_candidate in possible_catalogs:
            if catalog_candidate in image_to_catalog_assignment.values():
                continue 

        image_to_catalog_assignment[current] = catalog_candidate

        if geometrically_consistent(image_to_catalog_assignment, image_pairs_to_ang_dist, catalog_ang_dist, tolerance):
            result = DFS(image_to_catalog_assignment, image_stars, image_pairs_to_ang_dist, graph_hypotheses, catalog_ang_dist, tolerance)
            if result:
                return result

        del image_to_catalog_assignment[current]

    return None
        
    
#tape fix
def find_brightest_stars(image_path: str, num_stars_to_find: int):
    """
    Identifies the N brightest star-like objects using a robust, professional blob detection pipeline.
    """
    # --- 1. Load Image ---
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at '{image_path}'")
        return []
    
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- 2. Set up the Blob Detector with Professional Parameters ---
    
    # Create a parameter object
    params = cv2.SimpleBlobDetector_Params()

    # --- FILTER BY AREA ---
    # Filter out things that are too small (noise) or too big (text, galaxies)
    params.filterByArea = True
    params.minArea = 4       # Minimum area in pixels. Tune this.
    params.maxArea = 500     # Maximum area in pixels. Tune this.

    # --- FILTER BY CIRCULARITY ---
    # Filter out things that are not round (streaks, merged stars)
    # 1.0 is a perfect circle.
    params.filterByCircularity = True
    params.minCircularity = 0.5

    # --- FILTER BY CONVEXITY ---
    # Filter out things with "dents" (like our doughnut shape!)
    # 1.0 is perfectly convex.
    params.filterByConvexity = True
    params.minConvexity = 0.8

    # --- FILTER BY INERTIA RATIO ---
    # Filter out things that are elongated.
    # 1.0 is a circle, 0.0 is a line.
    params.filterByInertia = True
    params.minInertiaRatio = 0.8

    # Create a detector with the specified parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # --- 3. Detect Blobs ---
    # The detector works best on an inverted image where stars are black blobs on a white background.
    keypoints = detector.detect(255 - img_gray)

    if not keypoints:
        print("Warning: No blobs passed the detector's filters.")
        return []
        
    # The detector gives us keypoints. We want to sort them by their size (which correlates with brightness)
    # and then get their coordinates.
    
    # Sort keypoints by size (area) in descending order
    keypoints = sorted(keypoints, key=lambda k: k.size, reverse=True)
    
    # Extract coordinates from the top N keypoints
    brightest_stars_coords = [kp.pt for kp in keypoints[:num_stars_to_find]]
    
    print(f"Found {len(keypoints)} valid stars. Returning the {len(brightest_stars_coords)} brightest.")
    return brightest_stars_coords


# --- Example Usage ---
# (The visualization part remains the same)
if __name__ == "__main__":
    IMAGE_FILE = "testing3.png"
    NUM_STARS = 10
    
    star_coords = find_brightest_stars(IMAGE_FILE, NUM_STARS)
    
    if star_coords:
        print(f"\nCoordinates of the {len(star_coords)} brightest stars:")
        for i, (x, y) in enumerate(star_coords):
            print(f"  Star {i+1}: (x={x:.4f}, y={y:.4f})")
            
        img_color = cv2.imread(IMAGE_FILE)
        
        height, width, _ = img_color.shape
        cross_color = (0, 0, 255)
        cross_thickness = 1

        for i, (x, y) in enumerate(star_coords):
            ix, iy = int(round(x)), int(round(y))
            cv2.line(img_color, (ix, 0), (ix, height), cross_color, cross_thickness)
            cv2.line(img_color, (0, iy), (width, iy), cross_color, cross_thickness)
            cv2.circle(img_color, (ix, iy), 15, (0, 255, 0), 2)
            cv2.putText(img_color, str(i + 1), (ix + 18, iy + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
        unit_vectors = star_coords_to_unit_vector(star_coords, (CENTER_X, CENTER_Y), FOCAL_LENGTH)
        for (x, y, z) in unit_vectors:
            print(f"x = {x}, y = {y}, z = {z}")
        
        ang_dist = angular_distances2(star_coords)
        for (s1, s2), ang in ang_dist.items():
            print(f"Star pair: ({s1}, {s2}) -> Angle: {ang}")

        print("\n")
        # From unit vectors
        ang_dist = angular_distances(unit_vectors)
        for (s1, s2), ang in ang_dist.items():
            print(f"Star pair: ({s1}, {s2}) -> Angle: {ang}")

        output_filename = 'stars_identified.png'
        cv2.imwrite(output_filename, img_color)
        print(f"\nVisual result saved to '{output_filename}'.")
        
        cv2.imshow('Identified Stars', img_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #End of tape fix
        # Focal length 4.74 mm