import numpy as np
from scipy.spatial.distance import cdist
from ..core.coordinates import star_coords_to_unit_vector
from ..algorithms.quest import compute_attitude_quaternion
from ..core.attitude import rotational_angle_between_quaternions
from ..algorithms.refinement import refine_quaternion
from scipy.optimize import linear_sum_assignment

MAX_REFINEMENT_ITERATIONS = 20
QUATERNION_ANGLE_DIFFERENCE_THRESHOLD = 1e-3


def match_stars_hungarian(
    predicted_positions, detected_positions, distance_threshold=10.0
):

    if (
        predicted_positions is None
        or len(predicted_positions) == 0
        or detected_positions is None
        or len(detected_positions) == 0
    ):
        return []

    valid_predicted = []
    valid_detected = []
    
    for index, prev_coords in enumerate(predicted_positions):
        if (prev_coords is not None and 
            len(prev_coords) == 2 and 
            all(np.isfinite(coord) for coord in prev_coords)):
            valid_predicted.append((index, prev_coords))
    
    for index, detected_coords in enumerate(detected_positions):
        if (detected_coords is not None and 
            len(detected_coords) == 2 and 
            all(np.isfinite(coord) for coord in detected_coords)):
            valid_detected.append((index, detected_coords))

    if not valid_predicted or not valid_detected:
        return []

    predicted_indices, predicted_coords = zip(*valid_predicted)
    detected_indices, detected_coords = zip(*valid_detected)
    predicted_array = np.array(predicted_coords)
    detected_array = np.array(detected_coords)

    distances_matrix = cdist(predicted_array, detected_array)
    
    cost_matrix = np.where(
        distances_matrix <= distance_threshold, distances_matrix, 1e6
    )

    hungarian_pred_indices, hungarian_det_indices = linear_sum_assignment(cost_matrix)
    
    matches = []
    for filtered_pred_idx, filtered_det_idx in zip(hungarian_pred_indices, hungarian_det_indices):
        if cost_matrix[filtered_pred_idx, filtered_det_idx] < 1e6:
            original_pred_idx = predicted_indices[filtered_pred_idx]
            original_det_idx = detected_indices[filtered_det_idx]
            matches.append((original_pred_idx, original_det_idx))

    return matches



def track(
    previous_quaternion,
    previous_catalog_unit_vectors,
    previous_star_coords,
    detected_star_coords,
    camera_params,
    min_matches,
    distance_threshold=10.0
):
    if previous_quaternion is None or len(detected_star_coords) < min_matches:
        return previous_quaternion, [], [], [], {}

    # original_star_indices = list(previous_mapping.keys())
    # original_star_indices.sort()
    
    # sequential_to_original = {i: original_star_indices[i] for i in range(len(original_star_indices))}

    matches = match_stars_hungarian(
        previous_star_coords, detected_star_coords, distance_threshold
    )
    
    if len(matches) < min_matches:
        print("Not enough matches to track")
        return previous_quaternion, matches, [], [], {}

    matched_detected_pixels = [detected_star_coords[det_idx] for _, det_idx in matches]
    
    image_vectors = star_coords_to_unit_vector(
        matched_detected_pixels,
        center_coords=camera_params[1],
        f_x=camera_params[0][0],
        f_y=camera_params[0][1],
    )

    catalog_vectors = np.array([previous_catalog_unit_vectors[pred_idx] for pred_idx, _ in matches])
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
    
    used_catalog_vectors = [previous_catalog_unit_vectors[pred_idx] for pred_idx, _ in matches]
    used_star_coords = [detected_star_coords[det_idx] for _, det_idx in matches]
    
    # tracking_solution = {}
    # for pred_idx, det_idx in matches:
    #     if pred_idx in sequential_to_original:
    #         original_star_idx = sequential_to_original[pred_idx]
    #         star_id = previous_mapping[original_star_idx]
    #         tracking_solution[det_idx] = star_id
            
    return final_quaternion, matches, used_catalog_vectors, used_star_coords
