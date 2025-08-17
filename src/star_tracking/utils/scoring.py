TOLERANCE = 2


def score_solution(
    solution, image_angular_distances, catalog_angular_distances, tolerance=TOLERANCE
):
    """
    Function that scores a solution based on how well it describes observed angular distances in the image
    Inputs:
    - solution: dict where {star image ID: HIP ID}
    - image_angular_distances: dict where {(image star index 1, image star index 2): image angular distance}
    - catalog_angular_distances: dict where {(HIP ID 1, HIP ID 2): catalog angular distance}
    - tolerance: angular distance tolerance
    Outputs:
    - score: a float that describes the current solution's score. The perfect solution would have a score of 0
    """

    # Sum of total differences between image angular distance and catalog angular distance
    total_diff = 0
    # Count of stars that passed tolerance check
    count = 0

    for (s1, s2), img_ang_dist in image_angular_distances.items():

        if s1 in solution and s2 in solution:
            hip1, hip2 = solution[s1], solution[s2]

            cat_ang_dist = catalog_angular_distances.get(
                (hip1, hip2)
            ) or catalog_angular_distances.get((hip2, hip1))

            if cat_ang_dist is None:
                continue

            diff = abs(cat_ang_dist - img_ang_dist)
            # Account for cyclic warping to get correct diff
            diff = min(diff, 360 - diff)
            if diff > tolerance:
                continue
            total_diff += diff
            count += 1

    if count == 0:
        return float("inf")

    # coverage - the fraction of image stars mapped
    coverage = count / len(image_angular_distances)
    score = (total_diff / count) * (1 / coverage)
    return score


def load_solution_scoring(
    solutions, image_angular_distances, catalog_angular_distances
):
    """
    Function that scores all solutions
    Inputs:
    - solutions: array of possible solution dicts
    - image_angular_distances: dict where {(image star index 1, image star index 2): image angular distance}
    - catalog_angular_distances: dict where {(HIP ID 1, HIP ID 2): catalog angular distance}
    Outputs:
    - scored_solutions: array of tuples where the first element of each tuple is the solution
    and the second is it's score
    """
    return [
        (sol, score_solution(sol, image_angular_distances, catalog_angular_distances))
        for sol in solutions
    ]
