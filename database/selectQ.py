import sqlite3

# ==============================================================================
# --- USER CONFIGURATION ---
# ==============================================================================

# The name of the database file you want to inspect.
DATABASE_FILENAME = 'star_distances_sorted.db'

# The name of the table you want to query.
TABLE_NAME = 'AngularDistances'

# Margin parameters for the angular distance filter
MIN_DISTANCE = 0.1
MAX_DISTANCE = 0.5

# ==============================================================================

if __name__ == '__main__':
    
    connection = None
    try:
        # Connect to the database
        connection = sqlite3.connect(DATABASE_FILENAME)
        cursor = connection.cursor()

        print(f"--- Counting total rows in table '{TABLE_NAME}' ---")

        # Count all rows
        count_query = f"SELECT COUNT(*) FROM {TABLE_NAME};"
        cursor.execute(count_query)
        row_count = cursor.fetchone()[0]
        print(f"\nThe table '{TABLE_NAME}' contains {row_count:,} rows.")

        # ------------------------------------------------------------------------------
        # Query for star pairs within a given angular distance range
        # ------------------------------------------------------------------------------

        print(f"\n--- Querying star pairs with angular distance between {MIN_DISTANCE} and {MAX_DISTANCE} ---")

        distance_query = f"""
            SELECT hip1, hip2, angular_distance
            FROM {TABLE_NAME}
            WHERE angular_distance BETWEEN ? AND ?
        """
        cursor.execute(distance_query, (MIN_DISTANCE, MAX_DISTANCE))

        results = cursor.fetchall()

        # Create dictionary: key = (hip1, hip2), value = distance
        dict = { (row[0], row[1]): row[2] for row in results }

        print(f"Found {len(dict):,} pairs within the specified distance range.")
        
        # Optionally print first few entries
        print("\nSample entries:")
        for i, (key, value) in enumerate(dict.items()):
            print(f"{key} -> {value}")
            if i >= 4:
                break

    except sqlite3.Error as e:
        print(f"\n!!! DATABASE ERROR: {e}")
        print(f"!!! Please make sure the file '{DATABASE_FILENAME}' and table '{TABLE_NAME}' exist.")
        
    finally:
        if connection:
            connection.close()
            print("\n--- Connection closed. ---")
