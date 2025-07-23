import cv2
import numpy as np
import math

# The refine_centroid function is no longer needed, as the blob detector is more robust.
# We are replacing the entire logic.

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
    params.maxArea = 300     # Maximum area in pixels. Tune this.

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
    IMAGE_FILE = 'Ivostuff/polaris_fov.png'
    NUM_STARS = 3
    
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

        output_filename = 'stars_identified.png'
        cv2.imwrite(output_filename, img_color)
        print(f"\nVisual result saved to '{output_filename}'.")
        
        cv2.imshow('Identified Stars', img_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()