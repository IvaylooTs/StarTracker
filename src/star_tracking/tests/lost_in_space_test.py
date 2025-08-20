import os
from ..algorithms.lost_in_space import lost_in_space, tracking_helper
from ..core.attitude import rotational_angle_between_quaternions
from ..core.star_tracker import StarTracker, StarTrackingResult
import cProfile
import pstats

TEST_DIR = os.path.dirname(__file__)
IMAGE_FILE = os.path.join(TEST_DIR, "test_images", "testing8.png")
IMAGE_FILE_2 = os.path.join(TEST_DIR, "test_images", "testing9.png")
IMAGE_FILE_3 = os.path.join(TEST_DIR, "test_images", "testing10.png")
IMAGE_FILE_4 = os.path.join(TEST_DIR, "test_images", "testing11.png")
IMAGE_FILE_5 = os.path.join(TEST_DIR, "test_images", "testing12.png")
IMAGE_FILE_6 = os.path.join(TEST_DIR, "test_images", "testing13.png")
IMAGE_FILE_7 = os.path.join(TEST_DIR, "test_images", "testing14.png")
IMAGE_FILE_8 = os.path.join(TEST_DIR, "test_images", "testing15.png")
IMAGE_FILE_9 = os.path.join(TEST_DIR, "test_images", "testing16.png")
IMAGE_FILE_10 = os.path.join(TEST_DIR, "test_images", "testing17.png")
OUT_FILE = os.path.join(TEST_DIR, "out.prof")

if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    tracker = StarTracker(None, None)
    mode = "LIS"
    result = StarTrackingResult()
    image_files = [IMAGE_FILE, IMAGE_FILE_2, IMAGE_FILE_3, IMAGE_FILE_4, IMAGE_FILE_5, IMAGE_FILE_6, IMAGE_FILE_7, IMAGE_FILE_8, IMAGE_FILE_9, IMAGE_FILE_10]
    
    for idx, img in enumerate(image_files):
        print(f"\n--- Processing image {idx+1} ---")
        if mode == "LIS":
            result = tracker.lost_in_space(img)
            if result.success:
                print(f"Attitude: {result.quaternion}")
                print(f"Matched {result.num_matched_stars} stars")
                print(f"Best solution: {result.solution_mapping}")
                print(f"Processing time: {result.processing_time}")
                mode = "TRACKING"
                print("Switched to tracking mode")
        elif mode == "TRACKING":
            tracking_result = tracker.track_frame(img)
            if tracking_result.success:
                print(f"Attitude: {tracking_result.quaternion}")
                print(f"Matched {tracking_result.num_matched_stars} stars")
                print(f"Best solution: {tracking_result.solution_mapping}")
                print(f"Processing time: {tracking_result.processing_time}")
                angle = rotational_angle_between_quaternions(result.quaternion, tracking_result.quaternion)
                print(f"Rotational angle: {angle}")
            else:
                print(f"Tracking failed: {tracking_result.error_message}")
                mode = "LIS"
            
    
    # print("Lost-in-Space")
    # if result.success:
    #     print(f"Attitude: {result.quaternion}")
    #     print(f"Matched {result.num_matched_stars} stars")
    #     print(f"Best solution: {result.solution_mapping}")
    #     print(f"Processing time: {result.processing_time}")
        
    #     tracking_result = tracker.track_frame(IMAGE_FILE_2)
    #     print("Tracking")
    #     if tracking_result.success:
    #         print(f"Attitude: {tracking_result.quaternion}")
    #         print(f"Matched {tracking_result.num_matched_stars} stars")
    #         print(f"Best solution: {tracking_result.solution_mapping}")
    #         print(f"Processing time: {tracking_result.processing_time}")
    #         angle = rotational_angle_between_quaternions(result.quaternion, tracking_result.quaternion)
    #         print(f"Rotational angle: {angle}")
    #     else:
    #         print(f"Tracking failed: {tracking_result.error_message}")
    # else:
    #     print(f"Lost-in-space failed: {result.error_message}")
#     profiler.disable()
#     profiler.dump_stats(OUT_FILE)
#     stats = pstats.Stats(OUT_FILE).strip_dirs().sort_stats("cumulative")
# stats.print_stats(5)