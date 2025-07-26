import cv2
import numpy as np
# We are no longer using blob_log, so we can remove it.
# from skimage.feature import blob_log
import matplotlib.pyplot as plt

# =============================================================================
# --- CONFIGURABLE PARAMETERS ---
# =============================================================================

# --- Input ---
IMAGE_PATH = 'newFinal/clean1.jpg'

# --- Image Processing Parameters ---
N_STARS_TO_DETECT = 20

# --- NEW PARAMETERS FOR SHAPE FILTERING ---

# 1. Threshold for creating the binary image.
#    Any pixel brighter than this value (0-255) becomes a white pixel.
#    This is a critical parameter to tune. Start around 50-70 for this image.
BINARY_THRESHOLD = 60

# 2. Minimum area (in pixels) for an object to be considered a star.
#    This helps filter out single-pixel noise.
MIN_STAR_AREA = 2

# 3. Minimum circularity. 1.0 is a perfect circle.
#    This is our shape filter! It will reject elongated shapes like streaks and drawings.
#    A value of 0.7 is a good starting point to reject non-circular blobs.
MIN_CIRCULARITY = 0.8


# =============================================================================
# --- NEW CORE FUNCTION WITH SHAPE FILTER ---
# =============================================================================

def find_stars_with_shape_filter(image_path, n_stars, binary_threshold, min_area, min_circularity):
    """
    Finds stars by thresholding, finding contours, and filtering by shape (circularity)
    and size. This is much more robust against non-star objects.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # --- Step 1: Create a Binary Image ---
    # We apply a slight blur to reduce noise before thresholding. This makes contours smoother.
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
    # The cv2.threshold function returns the threshold value and the binary image.
    # We use _, binary_image because we don't need the returned threshold value.
    _, binary_image = cv2.threshold(blurred, binary_threshold, 255, cv2.THRESH_BINARY)
    
    # --- Step 2: Find Contours ---
    # cv2.findContours finds the outlines of all white objects in the binary image.
    # cv2.RETR_EXTERNAL means we only get the outermost contours (doesn't find holes).
    # cv2.CHAIN_APPROX_SIMPLE compresses the contour points to save memory.
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Initial detection: Found {len(contours)} potential objects (contours).")
    
    valid_stars = []

    # --- Step 3: Analyze Each Contour ---
    for contour in contours:
        # --- Filter by Area ---
        area = cv2.contourArea(contour)
        if area < min_area:
            continue # Skip this contour, it's too small (likely noise)

        # --- Filter by Shape (Circularity) ---
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue # Avoid division by zero for invalid shapes

        # This is the formula for circularity
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        
        if circularity < min_circularity:
            continue # Skip this contour, it's not round enough (it's a line or squiggle)

        # --- If it passes all filters, it's a valid star candidate ---
        # Calculate the center of the contour using "moments"
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue # Avoid division by zero
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # We store the centroid and its area (as a proxy for brightness)
        valid_stars.append(((cx, cy), area))

    print(f"Filtering complete: Found {len(valid_stars)} valid star-like objects.")
    
    # --- Step 4: Sort by Brightness/Size and Select the Top N ---
    # Sort the list of valid stars in descending order based on their area
    valid_stars.sort(key=lambda s: s[1], reverse=True)
    
    # Select the top N stars
    num_to_select = min(n_stars, len(valid_stars))
    brightest_stars = valid_stars[:num_to_select]
    
    # Extract just the centroids (the (x, y) coordinates)
    centroids = np.array([star[0] for star in brightest_stars])
    
    return centroids, image


def visualize_detections(image, centroids):
    """Displays the image with detected centroids circled."""
    # (This function remains unchanged)
    if image is None: return
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap='gray')
    plt.title(f'Detected {len(centroids)} Brightest & Roundest Stars')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=80, facecolors='none', edgecolors='r', linewidth=1.5)
    for i, (x, y) in enumerate(centroids):
        plt.text(x + 10, y + 10, str(i + 1), color='cyan', fontsize=12)
    plt.xlabel('X Pixel Coordinate')
    plt.ylabel('Y Pixel Coordinate')
    plt.show()

# =============================================================================
# --- MAIN EXECUTION ---
# =============================================================================
if __name__ == '__main__':
    try:
        # 1. Call our NEW function to detect and filter stars
        star_centroids, original_image = find_stars_with_shape_filter(
            IMAGE_PATH, N_STARS_TO_DETECT, BINARY_THRESHOLD, MIN_STAR_AREA, MIN_CIRCULARITY
        )
        
        # 2. Visualize the results
        if original_image is not None and len(star_centroids) > 0:
            print(f"\nFinal Result: Visualizing the top {len(star_centroids)} stars.")
            visualize_detections(original_image, star_centroids)
        else:
            print("No stars were detected that met the criteria.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")