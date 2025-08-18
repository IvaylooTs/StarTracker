import numpy as np
import math
import os
import matplotlib.pyplot as plt

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
RESULTS_FILE_PATH = "Testing/results3.txt"
BORESIGHT_BODY = np.array([0, 0, 1])
# ==============================================================================

def parse_filename_for_truth_radec(filename: str) -> tuple:
    """Parses the ground truth RA/Dec from the Stellarium filename format."""
    try:
        # Handle filenames like "...png001.png"
        if "png001.png" in filename:
            filename = filename.replace("png001.png", "png")

        # Strip everything after the first ".png"
        core_filename = filename.split('.png')[0]
        parts = core_filename.split('_')

        ra_part = parts[3]
        dec_part = parts[4]

        # Stellarium encodes p=., m=-
        ra_str = ra_part.replace('p', '.').replace('m', '-')
        dec_str = dec_part.replace('p', '.').replace('m', '-')

        return float(ra_str), float(dec_str)

    except (IndexError, ValueError):
        print(f"Warning: Could not parse RA/Dec from filename: '{filename}'")
        return None, None

def read_results_file_in_batches(filepath: str) -> list:
    """Reads the results file, parsing the filename, quaternion, and execution time into batches."""
    batches = []
    current_batch = []
    print(f"Reading and parsing results from: {filepath}")

    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()

                if line.startswith('---'):
                    if current_batch:
                        batches.append(current_batch)
                        current_batch = []
                    continue

                if (not line
                    or line.startswith('ImageName')):
                    continue  # skip headers and empty lines

                parts = line.split(';')

                # Expected format: filename ; q1 ; q2 ; q3 ; q4 ; time
                if len(parts) == 6:
                    filename = parts[0].strip()
                    true_ra, true_dec = parse_filename_for_truth_radec(filename)

                    if true_ra is None:
                        print(f"Skipping (bad RA/Dec): {filename}")
                        continue

                    try:
                        q = np.array([
                            float(parts[1].strip("()")),
                            float(parts[2].strip("()")),
                            float(parts[3].strip("()")),
                            float(parts[4].strip("()"))
                        ])

                        exec_time = float(parts[5].strip())

                        result = {
                            'filename': filename,
                            'true_ra_deg': true_ra,
                            'true_dec_deg': true_dec,
                            'quaternion': q,
                            'time': exec_time
                        }
                        current_batch.append(result)

                    except ValueError:
                        print(f"Skipping (parse error): {filename}")
                        continue
            # Add the last batch if the file doesn't end with a separator
            if current_batch:
                batches.append(current_batch)

    except FileNotFoundError:
        print(f"!!! FATAL ERROR: The results file was not found at '{filepath}'")

    return batches


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    norm = np.linalg.norm(q)
    if norm > 0:
        q = q / norm
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ])

def radec_to_unit_vector(ra_deg: float, dec_deg: float) -> np.ndarray:
    """Converts RA/Dec (in degrees) to a 3D inertial unit vector."""
    ra_rad = math.radians(ra_deg)
    dec_rad = math.radians(dec_deg)
    x = math.cos(dec_rad) * math.cos(ra_rad)
    y = math.cos(dec_rad) * math.sin(ra_rad)
    z = math.sin(dec_rad)
    return np.array([x, y, z])

def calculate_angular_error(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculates the angle (in degrees) between two 3D unit vectors."""
    dot_product = np.dot(v1, v2)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    return math.degrees(angle_rad)


# --- Main script execution ---

if __name__ == "__main__":
    results_batches = read_results_file_in_batches(RESULTS_FILE_PATH)

    if not results_batches:
        print("No valid results found to analyze.")
    else:
        print(f"\nFound {len(results_batches)} batches of results to validate.")

        all_pointing_errors = []
        all_execution_times = []
        all_batch_indices = []

        for i, batch in enumerate(results_batches):
            batch_num = i + 1
            print(f"\n--- Processing Batch {batch_num} ---")
            print("-" * 60)

            batch_pointing_errors = []
            batch_execution_times = []

            for result in batch:
                v_truth = radec_to_unit_vector(result['true_ra_deg'], result['true_dec_deg'])
                R = quat_to_rotation_matrix(result['quaternion'])
                v_predicted = R @ BORESIGHT_BODY
                error_deg = calculate_angular_error(v_truth, v_predicted)

                batch_pointing_errors.append(error_deg)
                batch_execution_times.append(result['time'])

                print(f"Image: {result['filename']:<55} | Error: {error_deg:.4f} degrees")

            if batch_pointing_errors:
                # --- Print Summary Statistics for the current batch ---
                average_error = np.mean(batch_pointing_errors)
                max_error = np.max(batch_pointing_errors)
                std_dev_error = np.std(batch_pointing_errors)
                average_time = np.mean(batch_execution_times)

                print("-" * 60)
                print(f"--- Batch {batch_num} Summary ---")
                print("=" * 35)
                print(f"Average Pointing Error: {average_error:.4f} degrees")
                print(f"Maximum Pointing Error: {max_error:.4f} degrees")
                print(f"Std. Deviation of Error: {std_dev_error:.4f} degrees")
                print(f"Average Execution Time: {average_time:.4f} seconds")
                print("=" * 35)

                # Append batch data to the overall lists for the final plot
                all_pointing_errors.extend(batch_pointing_errors)
                all_execution_times.extend(batch_execution_times)
                all_batch_indices.extend([batch_num] * len(batch_pointing_errors))

        if all_pointing_errors:
            # --- Print Overall Summary Statistics ---
            overall_average_error = np.mean(all_pointing_errors)
            overall_max_error = np.max(all_pointing_errors)
            overall_std_dev_error = np.std(all_pointing_errors)
            overall_average_time = np.mean(all_execution_times)
            
            print("\n--- Overall Validation Summary ---")
            print("=" * 35)
            print(f"Average Pointing Error: {overall_average_error:.4f} degrees")
            print(f"Maximum Pointing Error: {overall_max_error:.4f} degrees")
            print(f"Std. Deviation of Error: {overall_std_dev_error:.4f} degrees")
            print(f"Average Execution Time: {overall_average_time:.4f} seconds")
            print("=" * 35)

            # --- Plotting Block ---
            print("\nGenerating performance plot for all batches...")
            fig, ax = plt.subplots(figsize=(12, 7))

            # Use a colormap to automatically assign different colors to each batch
            scatter = ax.scatter(all_execution_times, all_pointing_errors, c=all_batch_indices,
                                 alpha=0.7, marker='o', s=15, cmap='viridis')

            ax.set_title('Performance: Pointing Error vs. Execution Time (All Batches)')
            ax.set_xlabel('Execution Time (seconds)')
            ax.set_ylabel('Pointing Error (degrees)')
            ax.grid(True)

            # Add a horizontal line for the overall average error
            ax.axhline(y=overall_average_error, color='b', linestyle='--', label=f'Overall Avg. Error: {overall_average_error:.4f}°')

            # Add a vertical line for the overall average time
            ax.axvline(x=overall_average_time, color='g', linestyle='--', label=f'Overall Avg. Time: {overall_average_time:.4f}s')

            # Create a legend for the scatter plot batches
            legend1 = ax.legend(*scatter.legend_elements(), title="Batches")
            ax.add_artist(legend1)
            
            ax.legend(loc='upper right')

            plt.show()