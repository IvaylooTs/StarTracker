import cv2
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# --- CONFIGURABLE PARAMETERS ---
# =============================================================================

# --- Input ---
IMAGE_PATH = './test_images/clean.jpg' 

# --- Initial Detection Parameters ---
N_STARS_TO_DETECT = 30     # The maximum number of stars to find.
BINARY_THRESHOLD = 87      # Pixel brightness cutoff (0-255). Keep this high to isolate bright objects.
MIN_STAR_AREA = 10         # The minimum number of pixels for an object to be considered a star.
MAX_STAR_AREA = 600        # The maximum pixel area. Filters out very large/bright objects.

# --- Shape Filter Gauntlet ---
MIN_CIRCULARITY = 0.80     # How close to a perfect circle (1.0). Good for filtering spiky noise.
MAX_ECCENTRICITY = 0.5     # Measures elongation. 0 is a perfect circle, closer to 1 is a line.
MIN_SOLIDITY = 0.98        # How "solid" the shape is. 1.0 has no holes or dents. Rejects notched shapes.

# --- Peak Brightness Filter ---
MIN_PEAK_RATIO = 0.9       # How much brighter the star's core must be than its immediate surroundings.

output_images = {}

def find_stars_with_advanced_filters(
    image_path: str, 
    n_stars: int, 
    binary_threshold: int = BINARY_THRESHOLD, 
    min_area: int = MIN_STAR_AREA, 
    max_area: int = MAX_STAR_AREA, 
    min_circularity: float = MIN_CIRCULARITY, 
    max_eccentricity: float = MAX_ECCENTRICITY, 
    min_solidity: float = MIN_SOLIDITY, 
    min_peak_ratio: float = MIN_PEAK_RATIO
    ):
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # --- Step 1: Find Initial Candidates ---
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary_image = cv2.threshold(blurred, binary_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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


    # --- Step 4: Sort by Brightness/Size and Select Top N ---
    valid_stars.sort(key=lambda s: s[1], reverse=True)
    brightest_stars = valid_stars[:n_stars]
    centroids = np.array([star[0] for star in brightest_stars]) if brightest_stars else np.array([])
    if output_images is not None:
        output_images['original'] = image
        output_images['binary'] = binary_image
    
    return centroids

def visualize_processing_steps(final_centroids, original_img = None, binary_img = None):
    if original_img is None:
        original_img = output_images.get("original")
    if binary_img is None:
        binary_img = output_images.get("binary")
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
    
def get_star_coords(image_path, n_stars):
    centroids = find_stars_with_advanced_filters(image_path, n_stars)
    return centroids

def display_star_detections(
    image_path: str, star_coords: list, output_filename: str = "stars_identified.png"
):
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

        cv2.circle(
            img_color, (x_int, y_int), circle_radius, circle_color, circle_thickness
        )
        cv2.drawMarker(
            img_color,
            (x_int, y_int),
            cross_color,
            markerType=cv2.MARKER_CROSS,
            thickness=cross_thickness,
        )

        text_position = (x_int + 10, y_int - 10)
        cv2.putText(
            img_color,
            str(idx),
            text_position,
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

    cv2.imwrite(output_filename, img_color)
    print(f"\nVisual result saved to '{output_filename}'.")

    cv2.imshow("Identified Stars", img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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

    return brightest_stars_coords
