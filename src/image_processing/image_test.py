import cv2
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# --- CONFIGURABLE PARAMETERS ---
# =============================================================================

# --- Input ---
# !!! IMPORTANT: Update this path to a valid image on your machine !!!
IMAGE_PATH = 'src\digital_twin\\test_images\\testing6.png' 

# --- Initial Detection Parameters ---
N_STARS_TO_DETECT = 10     # The maximum number of stars to find.
BINARY_THRESHOLD = 87      # Pixel brightness cutoff (0-255). Keep this high to isolate bright objects.
MIN_STAR_AREA = 20         # The minimum number of pixels for an object to be considered a star.
MAX_STAR_AREA = 500        # The maximum pixel area. Filters out very large/bright objects.

# --- Shape Filter Gauntlet ---
MIN_CIRCULARITY = 0.80     # How close to a perfect circle (1.0). Good for filtering spiky noise.
MAX_ECCENTRICITY = 0.5     # Measures elongation. 0 is a perfect circle, closer to 1 is a line.
MIN_SOLIDITY = 0.95        # How "solid" the shape is. 1.0 has no holes or dents. Rejects notched shapes.

# --- Peak Brightness Filter ---
MIN_PEAK_RATIO = 0.9       # How much brighter the star's core must be than its immediate surroundings.

# =============================================================================
# --- CORE DETECTION LOGIC ---
# =============================================================================

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
    Identifies the N brightest, most star-like objects in an image using a
    multi-stage filtering pipeline with Eccentricity and Solidity filters.

    Returns:
        A tuple containing: (centroids, original_image, binary_image)
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # --- Step 1: Find Initial Candidates ---
    # _, binary_image = cv2.threshold(blurred, binary_threshold, 255, cv2.THRESH_BINARY) ---- global treshhold version (old)
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    binary_image = cv2.adaptiveThreshold(
    blurred, 
    255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY, 
    blockSize=31, # Size of the neighborhood area (must be odd)
    C= -5        # A constant subtracted from the mean. A negative C detects darker objects.
)
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
            # The ellipse contains ((center_x, center_y), (minor_axis, major_axis), angle)
            (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
            if MA == 0: continue # Avoid division by zero for the eccentricity calculation
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

        # If a contour passes ALL shape tests, it's a good candidate.
        candidates_after_shape_filter.append(contour)

    print(f"After shape gauntlet (area, circularity, eccentricity, solidity): {len(candidates_after_shape_filter)} candidates remain.")

    valid_stars = []

    # --- Step 3: Peak Brightness Filter ---
    for contour in candidates_after_shape_filter:
        M = cv2.moments(contour)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        inner_radius, outer_radius = 2, 5

        # Create a "donut" mask to sample brightness around the star's core
        mask = np.zeros(image.shape, dtype=np.uint8)
        outer_mask = cv2.circle(mask.copy(), (cx, cy), outer_radius, 255, -1)
        inner_mask = cv2.circle(mask.copy(), (cx, cy), inner_radius, 255, -1)
        donut_mask = outer_mask & ~inner_mask

        try:
            # Calculate average brightness in the core and the surrounding ring
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

    return centroids, image, binary_image

# =============================================================================
# --- VISUALIZATION FUNCTION ---
# =============================================================================
def visualize_processing_steps(original_img, binary_img, final_centroids):
    """A helper function to visualize the detection steps using Matplotlib."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('1. Original Image')
    axes[0].axis('off')

    axes[1].imshow(binary_img, cmap='gray')
    axes[1].set_title('2. Processed View (Binary Threshold)')
    axes[1].axis('off')

    axes[2].imshow(original_img, cmap='gray')
    axes[2].set_title('3. Final Detections (Filtered)')
    if len(final_centroids) > 0:
        axes[2].scatter(final_centroids[:, 0], final_centroids[:, 1], s=120, facecolors='none', edgecolors='cyan', linewidth=2)
        for i, (x, y) in enumerate(final_centroids):
            axes[2].text(x + 12, y + 12, str(i + 1), color='magenta', fontsize=12, weight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# --- MAIN EXECUTION BLOCK ---
# This part of the script runs when you execute the file directly.
# =============================================================================
if __name__ == '__main__':
    try:
        # Call the main detection function with all the configured parameters
        star_centroids, original_image, binary_image = find_stars_with_advanced_filters(
            IMAGE_PATH, 
            N_STARS_TO_DETECT, 
            BINARY_THRESHOLD, 
            MIN_STAR_AREA, 
            MAX_STAR_AREA, 
            MIN_CIRCULARITY, 
            MAX_ECCENTRICITY, 
            MIN_SOLIDITY, 
            MIN_PEAK_RATIO
        )

        # --- Print and display the results ---
        if original_image is not None:
            print("\n--- Final Star Coordinates ---")
            
            if star_centroids.size > 0:
                coordinate_list = [tuple(coords) for coords in star_centroids]
                print(f"Successfully found {len(coordinate_list)} stars.")
                print("\nFormatted Coordinates (ordered by brightness):")
                for i, coords in enumerate(coordinate_list, 1):
                    print(f"  Star {i}: (x={coords[0]}, y={coords[1]})")
            else:
                print("No stars were detected that met all criteria.")

            # Call the visualization function to show the image plots
            print(f"\n--- Visualizing Results (Close the plot window to exit script) ---")
            visualize_processing_steps(original_image, binary_image, star_centroids)
        
        else:
            # This case should not be hit if the image exists, but it's good practice to have it.
            print("Processing failed: could not get a valid image.")

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Please make sure the IMAGE_PATH variable at the top of the script is correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")