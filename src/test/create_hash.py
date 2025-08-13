import sqlite3
import pickle
from collections import defaultdict
import time

# --- Configuration ---
# The path to your star catalog database
CATALOG_DB_PATH = 'src/test/star_distances_sorted.db' 

# The name of the output file that will be created
HASH_FILE_PATH = 'catalog_hash.pkl' 

# This is the "bin size" for our distances. It MUST match the tolerance
# used in the main script's acquisition stage. We set it to 2.0.
TOLERANCE = 1

def load_all_catalog_distances(db_path):
    """Loads all pre-calculated angular distances from the SQLite database."""
    print(f"Loading all angular distances from '{db_path}'...")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT hip1, hip2, angular_distance FROM AngularDistances")
        rows = cursor.fetchall()
        conn.close()
    except sqlite3.OperationalError as e:
        print(f"!!! DATABASE ERROR: Could not read from '{db_path}'. Make sure the file exists and is not corrupt.")
        print(f"Original error: {e}")
        return None

    print(f"Loaded {len(rows)} distance entries.")
    return rows

def build_and_save_hash():
    """
    Builds the geometric hash table for pairs and saves it to a file using pickle.
    """
    all_distances = load_all_catalog_distances(CATALOG_DB_PATH)

    if all_distances is None:
        print("Halting due to database error.")
        return

    print("Building the geometric hash table...")
    start_time = time.time()
    
    catalog_hash = defaultdict(list)

    # This is the core hashing logic
    for hip1, hip2, distance in all_distances:
        # Quantize the distance to create an integer key.
        # This groups similar distances into the same "bin".
        key = int(distance / TOLERANCE)
        
        # Add the pair of HIP IDs to the list for this key
        catalog_hash[key].append((hip1, hip2))

    end_time = time.time()
    print(f"Hash table built in {end_time - start_time:.2f} seconds.")
    print(f"The hash table has {len(catalog_hash)} unique keys (bins).")

    print(f"Saving hash table to '{HASH_FILE_PATH}'...")
    # Save the completed dictionary to a file. 'wb' means write in binary mode.
    with open(HASH_FILE_PATH, 'wb') as f:
        # pickle.dump is used to serialize a Python object into a file
        pickle.dump(dict(catalog_hash), f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"SUCCESS: The file '{HASH_FILE_PATH}' has been created.")
    print("You can now run your main script.")

# This makes the script runnable from the command line
if __name__ == "__main__":
    build_and_save_hash()