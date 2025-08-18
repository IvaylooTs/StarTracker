import os
import time
# Import your main tracker script as a module
import main_tracker 

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
IMAGE_FOLDER = "src/image_processing/test_images"
OUTPUT_FILE = "results2.txt"
VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

# ==============================================================================

def run_batch_processing():
    """
    Finds all images in the IMAGE_FOLDER, runs the star tracker on each one,
    and logs the results to the OUTPUT_FILE, including a final performance summary.
    """
    print(f"Starting batch processing for images in: '{IMAGE_FOLDER}'")
    
    try:
        image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(VALID_EXTENSIONS)]
    except FileNotFoundError:
        print(f"!!! ERROR: The folder '{IMAGE_FOLDER}' was not found. Please check the path.")
        return
        
    if not image_files:
        print("!!! No images found in the specified folder.")
        return

    print(f"Found {len(image_files)} images to process.")

    # <<< --- NEW: Initialize counters for the summary --- >>>
    total_execution_time = 0.0
    successful_runs = 0
    failed_runs = 0
    num = 10
    for i in range(9,10):
        with open(OUTPUT_FILE, 'a') as f:
            f.write("\n--- Batch Run Started ---\n")
            f.write("ImageName;Quaternion(w;x;y;z);ExecutionTime_s\n")

            for i, image_name in enumerate(image_files):
                full_image_path = os.path.join(IMAGE_FOLDER, image_name)
                print(f"\n--- Processing image {i+1}/{len(image_files)}: {image_name} ---")

                start_time = time.time()
                quaternion = main_tracker.lost_in_space(full_image_path, num)
                end_time = time.time()
                        
                execution_time = end_time - start_time
                        
                if quaternion is not None:
                    q_w, q_x, q_y, q_z = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
                    formatted_q = f"quaternioncoord({q_w:.4f};{q_x:.4f};{q_y:.4f};{q_z:.4f})"
                    status = "SUCCESS"
                            # <<< --- NEW: Update success counters --- >>>
                    successful_runs += 1
                    total_execution_time += execution_time
                else:
                    formatted_q = "FAILED"
                    status = "FAILED"
                            # <<< --- NEW: Update failure counter --- >>>
                    failed_runs += 1

                output_line = f"{image_name};{formatted_q};{execution_time:.4f}\n"
                f.write(output_line)
                print(f"Result for {image_name}: {status} | Time: {execution_time:.2f}s")
                    
             
            print(f"\nBatch processing complete. Results saved to '{OUTPUT_FILE}'.")
        num+=1
    # ==============================================================================
    # <<< --- NEW: Final Summary Block --- >>>
    # ==============================================================================
    print("\n" + "="*30)
    print("--- BATCH PROCESSING SUMMARY ---")
    print("="*30)
    print(f"Total Images Processed: {len(image_files)}")
    print(f"  - Successful Runs: {successful_runs}")
    print(f"  - Failed Runs:     {failed_runs}")
    
    # Calculate and print the average time, handling the case of zero successes
    if successful_runs > 0:
        average_time = total_execution_time / successful_runs
        print(f"\nAverage Execution Time (for successful runs): {average_time:.4f} seconds")
    else:
        print("\nNo runs were successful, cannot calculate average time.")
    print("="*30)
    # ==============================================================================

if __name__ == "__main__":
    run_batch_processing()