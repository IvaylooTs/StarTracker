import time

MIN_MATCHES = 10


def DFS(
    assignment,
    image_stars,
    candidate_hips,
    image_angular_distances,
    catalog_angular_distances,
    tolerance,
    solutions,
    start_time,
    max_time=15,
):
    """
    DFS to produce every possible mapping of image star to catalog star ID (even if not full mappings)
    Inputs:
    - assignment: a dict mapping stars in the image to candidate catalog stars (HIP IDs)
    - image_stars: list of stars detected in the image
    - candidate_hips: dict mapping each image star to a list of candidate catalog HIP IDs
    - image_angular_distances: dict where {(image star index 1, image star index 2): image angular distance}
    - catalog_angular_distances: dict where {(HIP ID 1, HIP ID 2): catalog angular distance}
    - tolerance: angular distance tolerance
    - solutions: a list to collect valid assignments (solutions) - initially empty
    """

    if time.time() - start_time == max_time:
        return

    # Does current assignment qualify as a solution
    if len(assignment) >= MIN_MATCHES:
        solutions.append(dict(assignment))

    # End of recursion condition if we've matched all stars
    if len(assignment) == len(image_stars):
        return

    # Selection of next star for assignment
    unassigned = [s for s in image_stars if s not in assignment]
    current_star = min(unassigned, key=lambda s: len(candidate_hips.get(s, [])))

    # Main loop for trying candidates for current star
    for hip_candidate in candidate_hips.get(current_star, []):

        # We don't want to assign the same HIP ID twice
        if hip_candidate in assignment.values():
            continue

        assignment[current_star] = hip_candidate

        # Check if the assignment is consistent and recurse to assign next star if so
        if is_consistent(
            assignment,
            image_angular_distances,
            catalog_angular_distances,
            tolerance,
            current_star,
        ):
            DFS(
                assignment,
                image_stars,
                candidate_hips,
                image_angular_distances,
                catalog_angular_distances,
                tolerance,
                solutions,
                start_time,
                max_time,
            )

        del assignment[current_star]


def is_consistent(
    assignment, image_angular_distances, catalog_angular_distances, tolerance, new_star
):
    """
    Bool function that checks if the current assignment is consistent with the angular distances from our image within a certain tolerance
    Inputs:
    - assignment: current assignment
    - image_angular_distances: dict where {(image star index 1, image star index 2): image angular distance}
    - catalog_angular_distances: dict where {(HIP ID 1, HIP ID 2): catalog angular distance}
    - tolerance: angular distance tolerance
    - new_star: index of new image star being added to the assignment
    """

    # Assignment doesn't have a pair of stars in it so is trivially consistent
    if len(assignment) < 2:
        return True

    # Iterate through all pairs of stars and their angular distances
    for (i1, i2), img_angle in image_angular_distances.items():

        # If the pair doesn't contain the star we're adding to the assignment we don't need to check it
        if new_star not in (i1, i2):
            continue

        # Check if pair is assigned HIP IDs
        if i1 in assignment and i2 in assignment:
            hip1 = assignment[i1]
            hip2 = assignment[i2]

            catalog_angle = catalog_angular_distances.get(
                (hip1, hip2)
            ) or catalog_angular_distances.get((hip2, hip1))

            if catalog_angle is None:
                return False

            if abs(catalog_angle - img_angle) > tolerance:
                return False
    return True

