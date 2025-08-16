import sqlite3
import pickle
from collections import defaultdict

DATABASE_PATH = "../database/star_distances_sorted.db"
HASH_FILE_PATH = "catalog_hash.pkl"

TOLERANCE = 2


def load_all_catalog_distances(db_path):
    """
    Loads all pre-calculated angular distances from the SQLite database.
    """

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT hip1, hip2, angular_distance FROM AngularDistances")
        rows = cursor.fetchall()
        conn.close()
    except sqlite3.OperationalError as e:
        print(
            f"!!! DATABASE ERROR: Could not read from '{db_path}'. Make sure the file exists and is not corrupt."
        )
        print(f"Original error: {e}")
        return None

    return rows


def build_and_save_hash(db_path):
    """
    Builds the geometric hash table for pairs and saves it to a file using pickle.
    """
    all_distances = load_all_catalog_distances(db_path)

    if all_distances is None:
        print("No distances exported from database, halting hashing process.")
        return

    catalog_hash = defaultdict(list)

    for hip1, hip2, distance in all_distances:

        key = int(distance / TOLERANCE)
        catalog_hash[key].append((hip1, hip2))

    with open(HASH_FILE_PATH, "wb") as f:

        pickle.dump(dict(catalog_hash), f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    build_and_save_hash(DATABASE_PATH)
