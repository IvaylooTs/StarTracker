import sqlite3
import numpy as np

def load_catalog_unit_vectors(db_path):
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT hip_id, x, y, z FROM Stars")

        hip_to_vector = {}
        for hip_id, x, y, z in cursor.fetchall():
            vector = np.array([x, y, z], dtype=float)
            hip_to_vector[hip_id] = vector

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return {}

    finally:
        if conn:
            conn.close()

    return hip_to_vector


def load_catalog_angular_distances(
    db_path="star_distances_sorted.db", table_name="AngularDistances"
):
    """
    Function that loads a dict from the database where {(HIP ID 1 [-], HIP ID 2 [-]): angular_distance [deg]}
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = f"SELECT hip1, hip2, angular_distance FROM {table_name}"
    cursor.execute(query)
    rows = cursor.fetchall()

    conn.close()

    return {(row[0], row[1]): row[2] for row in rows}
