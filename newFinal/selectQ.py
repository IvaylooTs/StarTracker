import sqlite3

# The name of the database file.
DATABASE_FILENAME = 'star_distances.db'

# How many rows to print before stopping.
ROW_LIMIT = 50

if __name__ == '__main__':
    
    connection = None
    try:
        # Connect to the database
        connection = sqlite3.connect(DATABASE_FILENAME)
        cursor = connection.cursor()

        print(f"--- Printing first {ROW_LIMIT} rows from 'AngularDistances' ---")
        print("Format: (hip1, hip2, angular_distance)")
        print("-" * 40)

        # The simplest possible SELECT query
        query = f"SELECT * FROM AngularDistances LIMIT {ROW_LIMIT};"
        
        # Execute the query and loop through the results
        for row in cursor.execute(query):
            print(row)

    except sqlite3.Error as e:
        print(f"\n!!! DATABASE ERROR: {e}")
        
    finally:
        # Make sure to close the connection
        if connection:
            connection.close()
            print("-" * 40)
            print("--- Connection closed. ---")