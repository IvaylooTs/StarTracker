import cv2
import numpy as np
import sqlite3
import math
from itertools import combinations
from itertools import product
from collections import defaultdict
from scipy.spatial.transform import Rotation as rotate
import sys
import os


IMAGE_PATH = './test_images/checker.png' 
import cv2
import cv2

def find_brightest_stars(image_path: str, num_stars_to_find: int, output_image_path: str = None):
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
    brightest_keypoints = keypoints[:num_stars_to_find]
    brightest_stars_coords = [kp.pt for kp in brightest_keypoints]

    # Draw the keypoints on the image
    output_img = cv2.drawKeypoints(
        img, brightest_keypoints, None,
        color=(0, 255, 0),  # green circles
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Add numbering labels next to the keypoints
    for i, kp in enumerate(brightest_keypoints, 1):
        x, y = int(kp.pt[0]), int(kp.pt[1])
        label = str(i)
        cv2.putText(
            output_img, label, (x + 5, y - 5),  # Slight offset to avoid overlap
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 255, 0),  # same green as circle
            thickness=1,
            lineType=cv2.LINE_AA
        )

    # Save the output image if path is provided
    if output_image_path:
        success = cv2.imwrite(output_image_path, output_img)
        if success:
            print(f"Saved output image with labels to '{output_image_path}'")
        else:
            print(f"Error: Failed to save output image to '{output_image_path}'")

    return brightest_stars_coords

if __name__ == "__main__":
    # Set parameters
    image_path = IMAGE_PATH
    n_stars_to_find = 5

    centroids = find_brightest_stars(image_path, n_stars_to_find, "stars_identified.png")
    stars = []
    stars_vectors = []
    for i, (x, y) in enumerate(centroids):
        print(f"Star {i + 1}: ({x:.2f}, {y:.2f})")
        stars.append((x,y))

    height = 768
    width = 1365
    angles = 78

    #optical center
    oc_height = height/2 
    oc_width = width/2 
    f = (height / 2)/ math.tan(math.radians(angles/2))
    print(f)
    for i, (x, y) in enumerate(centroids):
        stars_vectors.append(((x-oc_width)/f, (y-oc_height)/f,1))
        print(stars_vectors[i])

    for i in range(len(stars_vectors)):
        vec = stars_vectors[i]
        mag = math.sqrt(sum(c ** 2 for c in vec))
        norm_vec = [c / mag for c in vec]
        stars_vectors[i] = norm_vec

    for i in range(len(stars_vectors)):
        for j in range(i + 1, len(stars_vectors)):
            cosinus = sum(a * b for a, b in zip(stars_vectors[i], stars_vectors[j]))
            mag1 = math.sqrt(sum(a * a for a in stars_vectors[i]))
            mag2 = math.sqrt(sum(a * a for a in stars_vectors[j]))
            cos_theta = cosinus / (mag1 * mag2)
            cos_theta = max(-1.0, min(1.0, cos_theta))
            theta_rad = math.acos(cos_theta)
            theta_deg = math.degrees(theta_rad)
            print(i+1,j+1,theta_deg)