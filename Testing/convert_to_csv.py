import csv

# ==============================================================================
# --- CONFIGURATION ---
# ==============================================================================
# The input file you want to convert
INPUT_FILE_PATH = "results.txt" 
# The name of the CSV file that will be created
OUTPUT_FILE_PATH = "results_output.csv" 
# ==============================================================================

def parse_filename_for_truth_radec(filename: str) -> tuple:
    """Parses the ground truth RA/Dec from the specific Stellarium filename format."""
    try:
        core_filename = filename.split('.png')[0]
        parts = core_filename.split('_')
        # Expected format: Stellarium_test_image_RA_DEC
        ra_part, dec_part = parts[3], parts[4]
        ra_str = ra_part.replace('p', '.').replace('m', '-')
        dec_str = dec_part.replace('p', '.').replace('m', '-')
        return float(ra_str), float(dec_str)
    except (IndexError, ValueError):
        # Return None if the filename format is not as expected
        return None, None

def convert_log_to_csv(input_path: str, output_path: str):
    """
    Reads the semi-structured log file, parses the data, and writes it to a clean CSV file.
    """
    processed_data = []
    batch_run_id = 0
    line_number = 0

    print(f"Starting conversion of '{input_path}'...")

    try:
        with open(input_path, 'r') as infile:
            for line in infile:
                line_number += 1
                line = line.strip()

                # --- Handle Batch Headers ---
                if line.startswith('--- Batch Run Started ---'):
                    batch_run_id += 1
                    continue
                
                # --- Skip empty or header lines ---
                if not line or line.startswith('ImageName'):
                    continue

                # --- Parse the Data Line ---
                # Split the line into a maximum of 3 parts using the first two semicolons
                parts = line.split(';', 2)
                if len(parts) != 3:
                    print(f"  [Warning] Skipping malformed line #{line_number}: {line}")
                    continue
                
                image_name, content, time_str_raw = parts

                # --- Extract Ground Truth from Filename ---
                true_ra, true_dec = parse_filename_for_truth_radec(image_name)

                # --- Clean and Validate Execution Time ---
                try:
                    time_val = float(time_str_raw.strip().replace(',', '.'))
                except (ValueError, TypeError):
                    print(f"  [Warning] Skipping line #{line_number} due to invalid time value: '{time_str_raw}'")
                    continue

                # --- Determine Status and Parse Quaternion ---
                status = 'FAILED'
                q_w, q_x, q_y, q_z = '', '', '', '' # Default to empty strings for CSV

                if 'quaternioncoord' in content:
                    status = 'SUCCESS'
                    try:
                        numeric_part = content.replace('quaternioncoord(', '').replace(')', '')
                        coords = numeric_part.split(';')
                        if len(coords) == 4:
                            q_w, q_x, q_y, q_z = coords[0], coords[1], coords[2], coords[3]
                        else:
                            status = 'PARSE_ERROR'
                    except Exception:
                        status = 'PARSE_ERROR'
                
                # Append the structured data to our list
                processed_data.append([
                    batch_run_id,
                    image_name,
                    status,
                    true_ra if true_ra is not None else '',
                    true_dec if true_dec is not None else '',
                    time_val,
                    q_w, q_x, q_y, q_z
                ])

        print(f"Successfully processed {len(processed_data)} data entries.")

    except FileNotFoundError:
        print(f"!!! FATAL ERROR: The input file was not found at '{input_path}'")
        return

    # --- Write the Processed Data to a CSV File ---
    try:
        with open(output_path, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            
            # Write the header row
            writer.writerow([
                'BatchRunID', 'ImageName', 'Status', 'TrueRA', 'TrueDec',
                'ExecutionTime_s', 'QuaternionW', 'QuaternionX', 'QuaternionY', 'QuaternionZ'
            ])
            
            # Write all the data rows
            writer.writerows(processed_data)
        
        print(f"\nSuccessfully created CSV file: '{output_path}'")

    except IOError:
        print(f"!!! FATAL ERROR: Could not write to the file at '{output_path}'. Check permissions.")


# --- Main script execution ---
if __name__ == "__main__":
    convert_log_to_csv(INPUT_FILE_PATH, OUTPUT_FILE_PATH)