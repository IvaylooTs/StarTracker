from collections import defaultdict

# New hypotheses building functions

def generate_raw_votes(angular_distances, catalog_hash, tolerance):
    """
    Function that returns initial votes for each star index - hip id
    mapping based on the catalog hash pairs for our current bin
    Inputs:
    - angular_distances
    - catalog_hash: dict of pairs {bin number: [candidate pairs]}
    - tolerance: angular distance tolerance
    Outputs:
    - votes: dict {(image star index, HIP ID): number of votes}
    """

    votes = defaultdict(int)

    for (s1, s2), ang_dist in angular_distances.items():
        if ang_dist is None:
            continue
        key = int(ang_dist / tolerance)
        candidate_pairs = catalog_hash.get(key, [])
        for hip1, hip2 in candidate_pairs:
            votes[(s1, hip1)] += 1
            votes[(s1, hip2)] += 1
            votes[(s2, hip1)] += 1
            votes[(s2, hip2)] += 1

    return votes


def build_hypotheses_from_votes(raw_votes, min_votes=1):
    """
    Build hypotheses dict for DFS based on raw votes from generate_raw_votes function
    """
    hypotheses = defaultdict(set)
    for (star_idx, hip_id), vote_count in raw_votes.items():
        if vote_count >= min_votes:
            hypotheses[star_idx].add(hip_id)

    return hypotheses

# Old hypotheses building functions

def filter_catalog_angular_distances(cat_ang_dists, bounds):
    """
    Function that returns a dict where {(HIP ID 1 [-], HIP ID 2 [-]): angular_distance [deg]}
    angular_distance ∈ [min_ang_dist, max_ang_dist]
    Inputs:
    - cat_ang_dists: dict where {(HIP ID 1 [-], HIP ID 2 [-]): angular_distance [deg]}
    - bounds[2]: array where bounds[0] = min_ang_dist and bounds[1] = max_ang_dist
    """
    min_ang_dist, max_ang_dist = bounds
    filtered = {
        pair: ang_dist
        for pair, ang_dist in cat_ang_dists.items()
        if min_ang_dist <= ang_dist <= max_ang_dist
    }
    return filtered


def get_bounds(ang_dist, tolerance):
    """
    Function that returns an angular distance interval based on initial angular distance and a tolerance
    Outputs:
    - bounds: tuple where bounds[0] = min_ang_dist and bounds[1] = max_ang_dist
    """
    return (ang_dist - tolerance, ang_dist + tolerance)


def load_hypotheses(angular_distances, all_cat_ang_dists, tolerance, min_support):
    """
    Function that loads candidate catalog HIP IDs for each image star
    Inputs:
    - num_stars: number of stars from the image
    - angular_distances: dict where {(image star index 1, image star index 2): image angular distance}
    - all_cat_ang_dists: dict where {(HIP ID 1, HIP ID 2): catalog angular distance}
    - tolerance: angular distance tolerance
    - min_support: in how many pairs a certain HIP ID has to appear at minimum
    Outputs:
    - hypotheses_dict: {image star index: [HIP ID 1, ... HIP ID N]}
    """
    matches_counter = defaultdict(int)

    for (s1, s2), ang_dist in angular_distances.items():
        if ang_dist is None:
            continue

        bounds = get_bounds(ang_dist, tolerance)
        cur_cat_dict = filter_catalog_angular_distances(all_cat_ang_dists, bounds)

        for (hip1, hip2), _ in cur_cat_dict.items():
            matches_counter[(s1, hip1)] += 1
            matches_counter[(s1, hip2)] += 1
            matches_counter[(s2, hip1)] += 1
            matches_counter[(s2, hip2)] += 1

    hypotheses_dict = defaultdict(set)
    for (star_idx, hip_id), count in matches_counter.items():
        if count >= min_support:
            hypotheses_dict[star_idx].add(hip_id)

    return hypotheses_dict
