import numpy as np
from scipy.spatial.distance import cdist
from ..core.coordinates import star_coords_to_unit_vector
from ..algorithms.quest import compute_attitude_quaternion
from ..core.attitude import rotational_angle_between_quaternions
from ..algorithms.refinement import refine_quaternion

MAX_REFINEMENT_ITERATIONS = 20
QUATERNION_ANGLE_DIFFERENCE_THRESHOLD = 1e-3


def match_stars(predicted_positions, detected_positions, distance_threshold=10.0):
    """
    Match predicted star positions to detected star positions using Euclidean distance.
    Inputs:
    - predicted_positions (list of (x, y)): Projected star coords from previous attitude.
    - detected_positions (list of (x, y)): Star coords detected in current image.
    - distance_threshold (float): Max pixel distance for matching.
    Outputs:
    - matches: list of tuples (predicted_idx, detected_idx) of matched pairs.
    """

    valid_predicted = [
        (index, prev_coords)
        for index, prev_coords in enumerate(predicted_positions)
        if prev_coords is not None and len(prev_coords) == 2
    ]
    valid_detected = [
        (index, detected_coords)
        for index, detected_coords in enumerate(detected_positions)
        if detected_coords is not None and len(detected_coords) == 2
    ]

    predicted_indices, predicted_coords = zip(*valid_predicted)
    detected_indices, detected_coords = zip(*valid_detected)
    predicted_array = np.array(predicted_coords)
    detected_array = np.array(detected_coords)

    distances_matrix = cdist(predicted_array, detected_array)
    distances_matrix[distances_matrix > distance_threshold] = np.inf

    matches = []
    used_detected = set()

    for i, row in enumerate(distances_matrix):
        min_index = np.argmin(row)
        if row[min_index] != np.inf and min_index not in used_detected:
            matches.append((predicted_indices[i], detected_indices[min_index]))
            used_detected.add(min_index)

    return matches


def track(
    previous_quaternion,
    previous_catalog_unit_vectors,
    previous_star_coords,
    detected_star_coords,
    camera_params,
    distance_threshold=10.0,
):
    """
    Perform star tracking by matching predicted star positions to detected stars,
    then update the attitude quaternion.
    Inputs:
    - previous_quaternion: Current attitude quaternion estimate in [w, x, y, z] format.
    - previous_catalog_unit_vectors: Star unit vectors from catalog for previous frame.
    - detected_star_coords: Detected star pixel coords in current image.
    - camera_params: ((f_x, f_y), (c_x, c_y)) camera intrinsic parameters.
    - distance_threshold: Pixel distance threshold for matching.
    Outputs:
    - updated_quaternion: New attitude quaternion.
    - matches: List of matched star index pairs.
    """

    matches = match_stars(
        previous_star_coords, detected_star_coords, distance_threshold
    )
    if len(matches) < 3:
        print("Not enough matches to track")
        return previous_quaternion, matches

    matched_detected_pixels = [detected_star_coords[det_idx] for _, det_idx in matches]

    # Convert matched detected pixels to unit vectors
    image_vectors = star_coords_to_unit_vector(
        matched_detected_pixels,
        center_coords=camera_params[1],
        f_x=camera_params[0][0],
        f_y=camera_params[0][1],
    )

    # Extract matched catalog unit vectors using the correct indexing
    # The matches contain (previous_idx, detected_idx), where previous_idx
    # corresponds to the index in previous_catalog_unit_vectors
    catalog_vectors = np.array(
        [previous_catalog_unit_vectors[pred_idx] for pred_idx, _ in matches]
    )

    updated_quaternion = compute_attitude_quaternion(image_vectors, catalog_vectors)
    final_quaternion = updated_quaternion

    for _ in range(0, MAX_REFINEMENT_ITERATIONS):
        refined_quaternion = refine_quaternion(
            final_quaternion, catalog_vectors, image_vectors
        )
        delta_angle = rotational_angle_between_quaternions(
            final_quaternion, refined_quaternion
        )
        if delta_angle <= QUATERNION_ANGLE_DIFFERENCE_THRESHOLD:
            break
        final_quaternion = refined_quaternion

    return updated_quaternion, matches

