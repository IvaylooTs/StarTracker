import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.ndimage import center_of_mass
from typing import Tuple, List, Optional

# =============================================================================
# --- CONFIGURABLE PARAMETERS ---
# =============================================================================

# --- Default Parameters ---
N_STARS_TO_DETECT = 30     # The maximum number of stars to find.
BINARY_THRESHOLD = 87      # Pixel brightness cutoff (0-255). Keep this high to isolate bright objects.
MIN_STAR_AREA = 20         # The minimum number of pixels for an object to be considered a star.
MAX_STAR_AREA = 190        # The maximum pixel area. Filters out very large/bright objects.

# --- Shape Filter Gauntlet ---
MIN_CIRCULARITY = 0.70     # How close to a perfect circle (1.0). Good for filtering spiky noise.
MAX_ECCENTRICITY = 0.5     # Measures elongation. 0 is a perfect circle, closer to 1 is a line.
MIN_SOLIDITY = 0.95        # How "solid" the shape is. 1.0 has no holes or dents. Rejects notched shapes.

# --- Peak Brightness Filter ---
MIN_PEAK_RATIO = 0.9       # How much brighter the star's core must be than its immediate surroundings.
# --- Subpixel Parameters ---
SUBPIXEL_WINDOW_SIZE = 15  # Size of window around each star for subpixel refinement (must be odd)

# Global storage for output images (for visualization)
output_images = {}

# =============================================================================
# --- SUBPIXEL REFINEMENT FUNCTIONS ---
# =============================================================================

def is_sun_or_planet(image, cx, cy, radius=30):
    """Quick check if detection is sun/planet by looking at larger area"""
    h, w = image.shape
    if cx < radius or cy < radius or cx > w-radius or cy > h-radius:
        return False
    
    # Check if large area around object is uniformly bright (sun/planet characteristic)
    large_mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.circle(large_mask, (cx, cy), radius, 255, -1)
    large_area_pixels = image[large_mask == 255]
    
    if len(large_area_pixels) > 0:
        avg_brightness = np.mean(large_area_pixels)
        # If large area is very bright, it's likely sun/planet
        return avg_brightness > 120  # Adjust this threshold as needed
    
    return False

def gaussian_2d_fit(image_patch: np.ndarray, initial_centroid: Tuple[float, float]) -> Tuple[float, float]:
    """
    Fit a 2D Gaussian to the star and return the center with subpixel accuracy.
    Most accurate for well-behaved stellar PSFs but computationally intensive.
    """
    h, w = image_patch.shape
    y_indices, x_indices = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Initial guess parameters: [amplitude, x_center, y_center, sigma_x, sigma_y, background]
    amplitude_guess = np.max(image_patch) - np.min(image_patch)
    background_guess = np.min(image_patch)
    sigma_guess = 2.0
    
    initial_params = [
        amplitude_guess,
        initial_centroid[0],  # x center
        initial_centroid[1],  # y center
        sigma_guess,          # sigma_x
        sigma_guess,          # sigma_y
        background_guess      # background
    ]
    
    def gaussian_2d(params):
        amp, xc, yc, sigma_x, sigma_y, bg = params
        model = bg + amp * np.exp(-((x_indices - xc)**2 / (2*sigma_x**2) + (y_indices - yc)**2 / (2*sigma_y**2)))
        return model
    
    def objective(params):
        model = gaussian_2d(params)
        return np.sum((image_patch - model)**2)
    
    try:
        # Set bounds to keep parameters reasonable
        bounds = [
            (0, np.max(image_patch) * 2),      # amplitude
            (0, w-1),                           # x_center
            (0, h-1),                           # y_center
            (0.5, min(w, h)),                   # sigma_x
            (0.5, min(w, h)),                   # sigma_y
            (0, np.max(image_patch))            # background
        ]
        
        result = minimize(objective, initial_params, bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            return float(result.x[1]), float(result.x[2])  # x_center, y_center
        else:
            return initial_centroid
    except:
        return initial_centroid

def refine_centroid_subpixel(image: np.ndarray, pixel_centroid: Tuple[int, int], 
                           window_size: int, method: str = 'gaussian_fit') -> Tuple[float, float]:
    """
    Refine a pixel-level centroid to subpixel accuracy using the specified method.
    
    Args:
        image: The full grayscale image
        pixel_centroid: Initial (x, y) centroid at pixel level
        window_size: Size of the square window around the centroid (must be odd)
        method: Refinement method ('weighted_centroid', 'gaussian_fit', 'parabolic_fit', 'center_of_mass')
    
    Returns:
        Refined (x, y) centroid with subpixel accuracy
    """
    if window_size % 2 == 0:
        window_size += 1  # Ensure odd window size
        
    half_window = window_size // 2
    cx, cy = pixel_centroid
    h, w = image.shape
    
    # Define extraction bounds
    x_min = max(0, cx - half_window)
    x_max = min(w, cx + half_window + 1)
    y_min = max(0, cy - half_window)
    y_max = min(h, cy + half_window + 1)
    
    # Extract image patch
    image_patch = image[y_min:y_max, x_min:x_max].astype(np.float64)
    
    if image_patch.size == 0:
        return float(cx), float(cy)
    
    # Calculate initial centroid within the patch
    patch_cx = cx - x_min
    patch_cy = cy - y_min
    
    refined_cx, refined_cy = gaussian_2d_fit(image_patch, (patch_cx, patch_cy))
    
    # Convert back to full image coordinates
    final_cx = refined_cx + x_min
    final_cy = refined_cy + y_min
    
    return float(final_cx), float(final_cy)

# =============================================================================
# --- MAIN DETECTION FUNCTIONS ---
# =============================================================================

def find_stars_with_advanced_filters(
    image_path: str, 
    n_stars: int, 
    binary_threshold: int = BINARY_THRESHOLD, 
    min_area: int = MIN_STAR_AREA, 
    max_area: int = MAX_STAR_AREA, 
    min_circularity: float = MIN_CIRCULARITY, 
    max_eccentricity: float = MAX_ECCENTRICITY, 
    min_solidity: float = MIN_SOLIDITY, 
    min_peak_ratio: float = MIN_PEAK_RATIO,
    subpixel_method: str = 'gaussian_fit',
    subpixel_window_size: int = SUBPIXEL_WINDOW_SIZE
    ):
    """
    Enhanced star detection with optional subpixel accuracy refinement.
    
    Args:
        subpixel_method: If provided, enables subpixel refinement using specified method
                        ('weighted_centroid', 'gaussian_fit', 'parabolic_fit', 'center_of_mass')
                        If None, returns pixel-level coordinates for compatibility
    
    Returns:
        numpy array of star centroids with pixel or subpixel accuracy
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # --- Step 1: Find Initial Candidates ---
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    binary_image = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        blockSize=31,
        C=-5
    )
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
        if len(contour) >= 5:
            (x, y), (MA, ma), angle = cv2.fitEllipse(contour)
            if MA == 0: continue
            eccentricity = np.sqrt(1 - (min(MA, ma)**2 / max(MA, ma)**2))
            if eccentricity > max_eccentricity:
                continue
        else:
            continue

        # Filter 2d: Solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0: continue
        solidity = float(area) / hull_area
        if solidity < min_solidity:
            continue

        candidates_after_shape_filter.append(contour)

    valid_stars = []

    # --- Step 3: Peak Brightness Filter with Sun/Planet Rejection ---
    for contour in candidates_after_shape_filter:
        M = cv2.moments(contour)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # NEW: Quick sun/planet check
        if is_sun_or_planet(image, cx, cy):
            continue

        # NEW: Check original image brightness (reject false adaptive threshold detections)
        original_pixel_value = image[cy, cx]
        if original_pixel_value < 100:  # Raw pixel must be bright
            continue

        inner_radius, outer_radius = 2, 5

        mask = np.zeros(image.shape, dtype=np.uint8)
        outer_mask = cv2.circle(mask.copy(), (cx, cy), outer_radius, 255, -1)
        inner_mask = cv2.circle(mask.copy(), (cx, cy), inner_radius, 255, -1)
        donut_mask = outer_mask & ~inner_mask

        try:
            avg_inner_brightness = np.mean(image[inner_mask == 255])
            avg_outer_brightness = np.mean(image[donut_mask == 255])

            # NEW: Reject very faint detections (fixes black spot issue)
            if avg_inner_brightness < 50:  # Minimum brightness threshold
                continue

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
    
    # Store images for visualization
    if output_images is not None:
        output_images['original'] = image
        output_images['binary'] = binary_image
    
    if not brightest_stars:
        return np.array([])

    # --- Step 5: Optional Subpixel Refinement ---
    if subpixel_method is not None:
        subpixel_centroids = []
        for (cx, cy), area in brightest_stars:
            refined_cx, refined_cy = refine_centroid_subpixel(
                image, (cx, cy), subpixel_window_size, subpixel_method
            )
            subpixel_centroids.append([refined_cx, refined_cy])
        return np.array(subpixel_centroids)
    else:
        # Return pixel-level coordinates for backward compatibility
        centroids = np.array([star[0] for star in brightest_stars])
        return centroids

# =============================================================================
# --- VISUALIZATION FUNCTIONS ---
# =============================================================================

def visualize_processing_steps(final_centroids, original_img=None, binary_img=None):
    """A helper function to visualize the detection steps using Matplotlib."""
    if original_img is None:
        original_img = output_images.get("original")
    if binary_img is None:
        binary_img = output_images.get("binary")
    
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
        # Check if coordinates have decimal places (subpixel)
        is_subpixel = any(coord % 1 != 0 for coord in final_centroids.flatten())
        
        if is_subpixel:
            # Use + markers and show subpixel coordinates
            axes[2].scatter(final_centroids[:, 0], final_centroids[:, 1], 
                           marker='+', s=200, c='cyan', linewidth=1)
            axes[2].scatter(final_centroids[:, 0], final_centroids[:, 1], 
                           s=120, facecolors='none', edgecolors='red', linewidth=1)
            
            
        else:
            # Use original visualization for pixel coordinates
            axes[2].scatter(final_centroids[:, 0], final_centroids[:, 1], 
                           s=120, facecolors='none', edgecolors='cyan', linewidth=2)
            for i, (x, y) in enumerate(final_centroids):
                axes[2].text(x + 12, y + 12, str(i + 1), color='magenta', fontsize=12, weight='bold')
    
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def display_star_detections(
    image_path: str, star_coords: list, output_filename: str = "stars_identified.png"
):
    """
    # Display and save star detections with visual markers.
    # """
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

    # cv2.imshow("Identified Stars", img_color)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()