import numpy as np
import math
import matplotlib.pyplot as plt
from collections import defaultdict
import csv

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
# Updated file path to match the new CSV results file
RESULTS_FILE_PATH = "Testing/results.csv" 
# The number of data rows that constitute a single batch run
LINES_PER_BATCH = 155
BORESIGHT_BODY = np.array([0, 0, 1])
# ==============================================================================

def parse_filename_for_truth_radec(filename: str) -> tuple:
    """Parses the ground truth RA/Dec from the specific Stellarium filename format."""
    try:
        # Isolate the core part of the filename before any extensions
        core_filename = filename.split('.png')[0]
        parts = core_filename.split('_')
        # Extract the RA and Dec parts based on the known format
        ra_part, dec_part = parts[3], parts[4]
        # Convert 'p' to '.' and 'm' to '-' to create valid numbers
        ra_str = ra_part.replace('p', '.').replace('m', '-')
        dec_str = dec_part.replace('p', '.').replace('m', '-')
        return float(ra_str), float(dec_str)
    except (IndexError, ValueError):
        # Return None if the filename format is unexpected
        return None, None

def process_csv_results(filepath: str) -> dict:
    """
    Reads and processes the CSV results file, grouping data into batches.
    Fixes malformed filenames and skips bad rows safely.

    Args:
        filepath: The path to the CSV results file.

    Returns:
        A dictionary where keys are batch IDs and values are lists of run data.
    """
    batches = defaultdict(list)
    line_count = 0
    
    print("--- Stage 1: Loading and processing CSV data... ---")

    try:
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)  # skip header

            for row in reader:
                line_count += 1

                # Fix malformed filenames ending with .png001.png
                if len(row) >= 1 and row[0].endswith(".png001.png"):
                    row[0] = row[0].replace("001.png", ".png")

                # Ensure exactly 6 columns
                if len(row) != 6:
                    print(f"L{line_count+1}: FORMAT WARNING - Expected 6 columns, found {len(row)}. Skipping row: {row}")
                    continue

                image_name, qw, qx, qy, qz, time_str = row
                
                result = {
                    'quaternion': None,
                    'true_ra_deg': None,
                    'true_dec_deg': None,
                    'time': None
                }
                
                # Parse execution time
                try:
                    result['time'] = float(time_str)
                except (ValueError, TypeError):
                    print(f"L{line_count+1}: PARSE WARNING - Could not parse time '{time_str}'. Skipping entry.")
                    continue

                # Parse ground truth RA/Dec from the filename
                true_ra, true_dec = parse_filename_for_truth_radec(image_name)
                if true_ra is not None:
                    result['true_ra_deg'] = true_ra
                    result['true_dec_deg'] = true_dec
                else:
                    print(f"L{line_count+1}: PARSE WARNING - Could not get RA/Dec from filename '{image_name}'.")

                # Parse quaternion if available
                if qw and qx and qy and qz:
                    try:
                        q_values = [float(qw), float(qx), float(qy), float(qz)]
                        q = np.array(q_values)
                        norm = np.linalg.norm(q)
                        if norm > 1e-6:
                            result['quaternion'] = q / norm
                        else:
                            print(f"L{line_count+1}: PARSE ERROR - Quaternion has zero norm for '{image_name}'.")
                    except (ValueError, TypeError):
                        print(f"L{line_count+1}: PARSE ERROR - Could not parse quaternion values for '{image_name}'.")

                batch_id = (line_count - 1) // LINES_PER_BATCH + 1
                batches[batch_id].append(result)

    except FileNotFoundError:
        print(f"!!! FATAL ERROR: The results file was not found at '{filepath}'")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

    print(f"--- Stage 1 Complete: Processed {line_count} data entries into {len(batches)} batches. ---")
    return batches


def radec_to_unit_vector(ra_deg: float, dec_deg: float) -> np.ndarray:
    """Converts Right Ascension and Declination to a 3D unit vector."""
    ra_rad, dec_rad = math.radians(ra_deg), math.radians(dec_deg)
    return np.array([math.cos(dec_rad) * math.cos(ra_rad), 
                     math.cos(dec_rad) * math.sin(ra_rad), 
                     math.sin(dec_rad)])

def calculate_angular_error(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculates the angular separation between two unit vectors in degrees."""
    dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return math.degrees(np.arccos(dot_product))

def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Converts a quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
                     [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]])

# --- Main script execution ---
if __name__ == "__main__":
    
    all_batches = process_csv_results(RESULTS_FILE_PATH)
    
    if not all_batches:
        print("Analysis stopped: No valid data could be loaded or processed from the file.")
    else:
        print(f"\n--- Stage 2: Calculating final statistics... ---")
        
        summary_data = []
        all_successful_times = []

        for batch_id in sorted(all_batches.keys()):
            batch_runs = all_batches[batch_id]
            pointing_errors, execution_times = [], []
            successful_runs_count = 0
            
            for run in batch_runs:
                # A run is successful if it has a valid quaternion and ground truth
                if run['quaternion'] is not None and run['true_ra_deg'] is not None:
                    successful_runs_count += 1
                    execution_times.append(run['time'])
                    all_successful_times.append(run['time'])
                    
                    # Convert RA/Dec to a unit vector for error calculation
                    v_truth = radec_to_unit_vector(run['true_ra_deg'], run['true_dec_deg'])
                    # Convert result quaternion to rotation matrix
                    R = quat_to_rotation_matrix(run['quaternion'])
                    # Calculate the predicted pointing vector
                    v_predicted = R @ BORESIGHT_BODY
                    # Calculate the pointing error
                    error_deg = calculate_angular_error(v_truth, v_predicted)
                    if calculate_angular_error(v_truth, BORESIGHT_BODY) > 0:
                        error_av = abs((calculate_angular_error(v_truth, BORESIGHT_BODY) - error_deg)/calculate_angular_error(v_truth, BORESIGHT_BODY))
                        pointing_errors.append(error_av)
                    else: 
                        error_av = 0
                        pointing_errors.append(error_av)
            
            total_runs_in_batch = len(batch_runs)
            success_rate = (successful_runs_count / 155) * 100 
            avg_error = np.mean(pointing_errors) if successful_runs_count > 0 else 0
            avg_time = np.mean(execution_times) if successful_runs_count > 0 else 0
            summary_data.append({'id': batch_id, 'avg_error': avg_error, 'success_rate': success_rate, 'avg_time': avg_time})

        if summary_data and all_successful_times:
            print("\n--- Stage 3: Generating performance score plot... ---")
            
            grand_average_time = np.mean(all_successful_times)
            print(f"Grand Average Time across all successful runs: {grand_average_time:.4f}s")
            
            # Calculate a custom performance score for each batch
            for d in summary_data:
                c_accuracy = ( (d['avg_error'])) * 1
                c_speed = (1 / d['avg_time']) * 0.1 if d['avg_time'] > 0 else 0
                c_success = (d['success_rate'] / 100.0) * 1
                d['score'] = c_success + c_speed + c_accuracy
            
            summary_data.sort(key=lambda x: x['id'])
            
            ids = [f"{d['id']+3} Stars" for d in summary_data]
            scores = [d['score'] for d in summary_data]
            
            fig, ax = plt.subplots(figsize=(12, 7))
            best_score_val = max(scores) if scores else 0
            colors = ['gold' if s == best_score_val else 'purple' for s in scores]
            
            ax.plot(ids, scores, color='red', marker='o', linestyle='-', alpha=0.8)
            ax.set_xlabel('Number of stars')
            ax.set_ylabel('Performance Score')
            ax.set_title('Overall Performance Score by Batch Run')
            ax.grid(True, axis='y', linestyle=':', alpha=0.7)
            
            # Add score labels on top of each bar
            
            
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            
            output_filename = "performance_by_run.png"
            plt.savefig(output_filename)
            print(f"\nPlot saved successfully as '{output_filename}'")
            plt.show()

        else:
            print("\nAnalysis complete: No successful runs were found to create a performance score plot.")