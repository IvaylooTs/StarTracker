import cv2
import numpy as np
import matplotlib.pyplot as plt
# We are no longer using the complex Gaussian fit, so we can remove scipy
# from scipy.optimize import curve_fit

# =============================================================================
# --- CONFIGURABLE PARAMETERS ---
# =============================================================================

# --- Input ---
IMAGE_PATH = 'src/image_processing_test/test_images/clean.jpg' # The image with drawings

# --- Initial Detection Parameters ---
N_STARS_TO_DETECT = 20
BINARY_THRESHOLD = 87      # Keep this high to isolate bright objects
MIN_STAR_AREA = 2
MAX_STAR_AREA = 150        # Filter out very large objects

# --- Shape Filter ---
MIN_CIRCULARITY = 0.85      # Keep this high to ensure roundness

# --- NEW: PEAK BRIGHTNESS FILTER ---
# This is the ratio of brightness between the core and the surrounding area.
# A value of 1.2 means the center must be at least 20% brighter than its surroundings.
# This will reject plateaus and objects with holes.
MIN_PEAK_RATIO = 0.9

# =============================================================================
# --- CORE DETECTION FUNCTION (REVISED WITH NEW FILTER) ---
# =============================================================================

def find_stars_with_peak_filter(image_path, n_stars, binary_threshold, min_area, max_area, min_circularity, min_peak_ratio):
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # --- Step 1: Find Candidates with Contour Analysis ---
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary_image = cv2.threshold(blurred, binary_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Initial detection: Found {len(contours)} potential objects.")
    
    candidates_after_shape_filter = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (min_area < area < max_area):
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0: continue
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity < min_circularity:
            continue
            
        candidates_after_shape_filter.append(contour)

    print(f"After shape & area filtering: {len(candidates_after_shape_filter)} candidates remain.")

    valid_stars = []
    
    # --- Step 2: NEW - PEAK BRIGHTNESS FILTER ---
    for contour in candidates_after_shape_filter:
        # Calculate centroid using moments
        M = cv2.moments(contour)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Define radii for our inner core and outer ring
        inner_radius = 2
        outer_radius = 4

        # Create two masks: one for the inner core, one for the outer area
        mask = np.zeros(image.shape, dtype=np.uint8)
        outer_mask = cv2.circle(mask.copy(), (cx, cy), outer_radius, 255, -1)
        inner_mask = cv2.circle(mask.copy(), (cx, cy), inner_radius, 255, -1)
        
        # The "donut" is the outer circle minus the inner circle
        donut_mask = outer_mask - inner_mask

        try:
            # Calculate the average brightness in the core and the donut
            # We use the ORIGINAL grayscale image for this
            avg_inner_brightness = np.mean(image[inner_mask == 255])
            avg_outer_brightness = np.mean(image[donut_mask == 255])
            
            # Avoid division by zero if the outer ring is completely black
            if avg_outer_brightness == 0:
                continue

            # This is the core of the filter
            brightness_ratio = avg_inner_brightness / avg_outer_brightness

            # If the center is significantly brighter than the surroundings, it's a star
            if brightness_ratio > min_peak_ratio:
                valid_stars.append(((cx, cy), M["m00"]))

        except (ValueError, ZeroDivisionError):
            # This can happen for very small or edge-case contours
            continue
            
    print(f"After peak brightness filtering: {len(valid_stars)} valid stars remain.")
    
    # --- Step 3: Sort by Brightness/Size and Select Top N ---
    valid_stars.sort(key=lambda s: s[1], reverse=True)
    num_to_select = min(n_stars, len(valid_stars))
    brightest_stars = valid_stars[:num_to_select]
    centroids = np.array([star[0] for star in brightest_stars])
    
    return centroids, image, binary_image


# =============================================================================
# --- VISUALIZATION FUNCTION (UNCHANGED) ---
# =============================================================================

def visualize_processing_steps(original_img, binary_img, final_centroids):
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
# --- MAIN EXECUTION (UPDATED) ---
# =============================================================================
if __name__ == '__main__':
    try:
        # --- Step 1: Run the detection and get the results ---
        star_centroids, original_image, binary_image = find_stars_with_peak_filter(
            IMAGE_PATH, N_STARS_TO_DETECT, BINARY_THRESHOLD, MIN_STAR_AREA, MAX_STAR_AREA, MIN_CIRCULARITY, MIN_PEAK_RATIO
        )
        
        # --- Step 2: Print all coordinate information to the console FIRST ---
        if original_image is not None:
            print("\n--- Final Star Coordinates ---")
            
            if star_centroids.size > 0:
                # Convert the NumPy array to a list of tuples
                coordinate_list = [tuple(coords) for coords in star_centroids]
                
                print(f"Successfully found {len(coordinate_list)} stars.")
                
                # Print the formatted coordinates for each star
                print("\nFormatted Coordinates (ordered by brightness):")
                for i, coords in enumerate(coordinate_list, 1):
                    print(f"  Star {i}: (x={coords[0]}, y={coords[1]})")
            else:
                print("No stars were detected.")

            # --- Step 3: Display the visualization window LAST ---
            # The script will now pause here until you close the image.
            print(f"\n--- Visualizing Results (Close window to exit script) ---")
            visualize_processing_steps(original_image, binary_image, star_centroids)
        
        else:
            print("No stars were detected that met all criteria.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")