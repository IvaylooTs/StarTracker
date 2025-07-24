import sqlite3
import time

# ==============================================================================
# --- USER CONFIGURATION ---
# ==============================================================================

# The name of the database file you want to modify.
DATABASE_FILENAME = 'star_distances.db'

# The threshold for deletion.
# Any row where 'angular_distance' is GREATER THAN this value will be deleted.
DISTANCE_THRESHOLD = 77

# ==============================================================================

if __name__ == '__main__':
    print("--- Database Deletion Script ---")
    print("!!! WARNING: This script will permanently delete data. !!!")
    print(f"!!! Make sure you have a backup of '{DATABASE_FILENAME}'. !!!")
    
    # Give the user a moment to cancel if they ran it by accident
    try:
        input("\nPress Enter to continue, or Ctrl+C to cancel...")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        exit()

    connection = None  # Initialize connection to None
    try:
        # Connect to the database. This connection is read/write by default.
        connection = sqlite3.connect(DATABASE_FILENAME)
        cursor = connection.cursor()

        # --- Step 1: Report the state BEFORE the deletion ---
        print("\n--- Analyzing current state of the database... ---")
        cursor.execute("SELECT COUNT(*) FROM AngularDistances;")
        initial_row_count = cursor.fetchone()[0]
        print(f"Total rows in 'AngularDistances' before deletion: {initial_row_count:,}")

        # --- Step 2: Perform the DELETE operation ---
        print(f"\n--- Deleting rows where angular_distance > {DISTANCE_THRESHOLD}... ---")
        start_time = time.time()

        # The SQL query to delete rows based on the condition.
        # We use a placeholder (?) for the value to prevent SQL injection issues.
        delete_query = "DELETE FROM AngularDistances WHERE angular_distance > ?;"
        cursor.execute(delete_query, (DISTANCE_THRESHOLD,))
        
        # After executing, cursor.rowcount holds the number of deleted rows
        num_deleted_rows = cursor.rowcount

        # --- Step 3: Commit the changes to the database ---
        # This is the most crucial step. Without this, the changes are not saved!
        print("Committing changes to the database file...")
        connection.commit()
        
        end_time = time.time()
        print(f"Deletion and commit took {end_time - start_time:.2f} seconds.")

        # --- Step 4: Report the state AFTER the deletion ---
        print("\n--- Reporting final state... ---")
        print(f"Number of rows deleted: {num_deleted_rows:,}")
        
        cursor.execute("SELECT COUNT(*) FROM AngularDistances;")
        final_row_count = cursor.fetchone()[0]
        print(f"Total rows in 'AngularDistances' after deletion: {final_row_count:,}")

    except sqlite3.Error as e:
        print(f"\n!!! DATABASE ERROR: {e}")
        if connection:
            print("An error occurred. Rolling back any changes.")
            connection.rollback() # Undo any partial changes if an error happened
            
    finally:
        # --- Step 5: Always close the connection ---
        if connection:
            connection.close()
            print("\n--- Connection closed. Operation complete. ---")
