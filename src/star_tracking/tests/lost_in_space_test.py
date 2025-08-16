from ..algorithms.lost_in_space import lost_in_space

import os
from ..algorithms.lost_in_space import lost_in_space

TEST_DIR = os.path.dirname(__file__)
IMAGE_FILE = os.path.join(TEST_DIR, "test_images", "testing1.png")

if __name__ == "__main__":
    result = lost_in_space(IMAGE_FILE)
    if result.success:
        print(f"Attitude: {result.quaternion}")
        print(f"Matched {result.num_matched_stars} stars")
        print(f"Best solution: {result.solution_mapping}")
    else:
        print(f"Failed: {result.error_message}")