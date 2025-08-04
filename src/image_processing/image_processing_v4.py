import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_PATH = "./test_images/clean1.jpg" 

N_STARS_TO_DETECT = 30
BINARY_THRESHOLD = 87      
MIN_STAR_AREA = 5
MAX_STAR_AREA = 450     

MIN_CIRCULARITY = 0.60      
MIN_PEAK_RATIO = 0.9

output_images = {}

def find_stars_with_peak_filter(image_path, n_stars, binary_threshold = BINARY_THRESHOLD, min_area = MIN_STAR_AREA, max_area = MAX_STAR_AREA, min_circularity = MIN_CIRCULARITY, min_peak_ratio = MIN_PEAK_RATIO):
    global output_images
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

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
    
    for contour in candidates_after_shape_filter:
        M = cv2.moments(contour)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        inner_radius = 2
        outer_radius = 4

        mask = np.zeros(image.shape, dtype=np.uint8)
        outer_mask = cv2.circle(mask.copy(), (cx, cy), outer_radius, 255, -1)
        inner_mask = cv2.circle(mask.copy(), (cx, cy), inner_radius, 255, -1)
        
        donut_mask = outer_mask - inner_mask

        try:
            avg_inner_brightness = np.mean(image[inner_mask == 255])
            avg_outer_brightness = np.mean(image[donut_mask == 255])
            
            if avg_outer_brightness == 0:
                continue

            brightness_ratio = avg_inner_brightness / avg_outer_brightness

            if brightness_ratio > min_peak_ratio:
                valid_stars.append(((cx, cy), M["m00"]))

        except (ValueError, ZeroDivisionError):
            continue
            
    print(f"After peak brightness filtering: {len(valid_stars)} valid stars remain.")
    
    valid_stars.sort(key=lambda s: s[1], reverse=True)
    num_to_select = min(n_stars, len(valid_stars))
    brightest_stars = valid_stars[:num_to_select]
    centroids = np.array([star[0] for star in brightest_stars])
    
    if output_images is not None:
        output_images['original'] = image
        output_images['binary'] = binary_image
    
    return centroids
    
def visualize_processing_steps(final_centroids, original_img=None, binary_img=None):
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
        axes[2].scatter(final_centroids[:, 0], final_centroids[:, 1], s=120, facecolors='none', edgecolors='cyan', linewidth=2)
        for i, (x, y) in enumerate(final_centroids):
            axes[2].text(x + 12, y + 12, str(i + 1), color='magenta', fontsize=12, weight='bold')
    axes[2].axis('off')
    plt.tight_layout()
    plt.show()

def get_star_coords(image_path, n_stars):
    centroids = find_stars_with_peak_filter(image_path, n_stars)
    return centroids

