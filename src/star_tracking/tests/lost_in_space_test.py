import os
from ..algorithms.lost_in_space import lost_in_space, tracking_helper
from ..core.attitude import rotational_angle_between_quaternions
from ..core.star_tracker import StarTracker

TEST_DIR = os.path.dirname(__file__)
IMAGE_FILE = os.path.join(TEST_DIR, "test_images", "testing4.png")
IMAGE_FILE_2 = os.path.join(TEST_DIR, "test_images", "testing5.png")

if __name__ == "__main__":
    tracker = StarTracker(None, None)
    result = tracker.lost_in_space(IMAGE_FILE)
    
    print("Lost-in-Space")
    if result.success:
        print(f"Attitude: {result.quaternion}")
        print(f"Matched {result.num_matched_stars} stars")
        print(f"Best solution: {result.solution_mapping}")
        
        tracking_result = tracker.track_frame(IMAGE_FILE_2)
        print("Tracking")
        if tracking_result.success:
            print(f"Attitude: {tracking_result.quaternion}")
            print(f"Matched {tracking_result.num_matched_stars} stars")
            print(f"Best solution: {tracking_result.solution_mapping}")
            angle = rotational_angle_between_quaternions(result.quaternion, tracking_result.quaternion)
            print(f"Rotational angle: {angle}")
        else:
            print(f"Tracking failed: {tracking_result.error_message}")
    else:
        print(f"Lost-in-space failed: {result.error_message}")