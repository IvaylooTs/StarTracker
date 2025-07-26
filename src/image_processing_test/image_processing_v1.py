import cv2
import numpy as np
from skimage.feature import blob_log
import matplotlib.pyplot as plt
import itertools
import sqlite3
import os
from collections import defaultdict

# =============================================================================
# --- CONFIGURABLE PARAMETERS ---
# =============================================================================

# --- Input & Database ---
IMAGE_PATH = 'newFinal/clean3.jpg' # Or your 'clean.jpg'
DATABASE_PATH = 'star_distances.db'

# --- Image Processing Parameters ---
N_STARS_TO_DETECT = 20

# --- Star Detection Parameters ---
MIN_SIGMA = 20
MAX_SIGMA = 50
THRESHOLD = 0.05

# =============================================================================
# --- CORE FUNCTIONS ---
# (No changes needed in any of these functions)
# =============================================================================

def find_star_centroids(image_path, n_stars, min_sigma, max_sigma, threshold):
    # (Unchanged)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    image_normalized = image / 255.0
    blobs = blob_log(image_normalized, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=10, threshold=threshold)
    if len(blobs) == 0:
        print("Warning: No stars detected.")
        return np.array([]), None
    blobs_sorted_by_brightness = blobs[blobs[:, 2].argsort()[::-1]]
    num_to_select = min(n_stars, len(blobs_sorted_by_brightness))
    brightest_blobs = blobs_sorted_by_brightness[:num_to_select]
    centroids = brightest_blobs[:, [1, 0]]
    print(f"Successfully detected {len(centroids)} stars.")
    return centroids, image


def visualize_detections(image, centroids):
    # (Unchanged)
    if image is None: return
    plt.figure(figsize=(10, 8))
    plt.imshow(image, cmap='gray')
    plt.title(f'Detected {len(centroids)} Brightest Stars')
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
        # 1. Detect Stars and get Centroids
        star_centroids, original_image = find_star_centroids(
            IMAGE_PATH, N_STARS_TO_DETECT, MIN_SIGMA, MAX_SIGMA, THRESHOLD
        )
        visualize_detections(original_image, star_centroids)

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")