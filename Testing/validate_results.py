import numpy as np
import math
import os
import matplotlib.pyplot as plt

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
RESULTS_FILE_PATH = "Testing/results2.txt"
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

def read_results_file(filepath: str) -> list:
    """Reads the results file, parsing the filename, quaternion, and execution time."""
    valid_results = []
    print(f"Reading and parsing results from: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if (not line 
                    or line.startswith('---') 
                    or line.startswith('ImageName')):
                    continue  # skip headers and separators
                
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
                        valid_results.append(result)

                    except ValueError:
                        print(f"Skipping (parse error): {filename}")
                        continue
                            
    except FileNotFoundError:
        print(f"!!! FATAL ERROR: The results file was not found at '{filepath}'")
        
    return valid_results


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
    results_data = read_results_file(RESULTS_FILE_PATH)
    
    if not results_data:
        print("No valid results found to analyze.")
    else:
        print(f"\nFound {len(results_data)} successful runs to validate.")
        
        pointing_errors = []
        execution_times = []
        
        print("\n--- Attitude Pointing Error Report ---")
        print("-" * 60)
        
        for result in results_data:
            v_truth = radec_to_unit_vector(result['true_ra_deg'], result['true_dec_deg'])
            R = quat_to_rotation_matrix(result['quaternion'])
            v_predicted = R @ BORESIGHT_BODY
            error_deg = calculate_angular_error(v_truth, v_predicted)
            
            pointing_errors.append(error_deg)
            execution_times.append(result['time'])
            
            print(f"Image: {result['filename']:<55} | Error: {error_deg:.4f} degrees")
        
        if pointing_errors:
            # --- Print Summary Statistics ---
            average_error = np.mean(pointing_errors)
            max_error = np.max(pointing_errors)
            std_dev_error = np.std(pointing_errors)
            # <<< --- NEW: Calculate average time --- >>>
            average_time = np.mean(execution_times)
            
            print("-" * 60)
            print("\n--- Validation Summary ---")
            print("=" * 35)
            print(f"Average Pointing Error: {average_error:.4f} degrees")
            print(f"Maximum Pointing Error: {max_error:.4f} degrees")
            print(f"Std. Deviation of Error: {std_dev_error:.4f} degrees")
            # <<< --- NEW: Print average time --- >>>
            print(f"Average Execution Time: {average_time:.4f} seconds")
            print("=" * 35)
            
            # --- Plotting Block ---
            print("\nGenerating performance plot...")
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.scatter(execution_times, pointing_errors, alpha=0.7, edgecolors='r', c='red', marker='o', s=11)
            
            ax.set_title('Performance: Pointing Error vs. Execution Time')
            ax.set_xlabel('Execution Time (seconds)')
            ax.set_ylabel('Pointing Error (degrees)')
            ax.grid(True)
            
            # Add a horizontal line for the average error
            ax.axhline(y=average_error, color='b', linestyle='--', label=f'Average Error: {average_error:.4f}°')
            
            # <<< --- NEW: Add a vertical line for the average time --- >>>
            ax.axvline(x=average_time, color='g', linestyle='--', label=f'Average Time: {average_time:.4f}s')

            ax.legend()
            
            plt.show()