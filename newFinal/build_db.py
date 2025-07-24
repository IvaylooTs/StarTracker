import pandas as pd
import numpy as np
import sqlite3
import itertools
from tqdm import tqdm
import time

# ==============================================================================
# --- USER CONFIGURATION ---
# ==============================================================================

# 1. The name of your Hipparcos CSV catalog file.
CATALOG_FILENAME = 'newFinal/hipparcos-voidmain.csv'

# 2. The desired name for the output database file.
DATABASE_FILENAME = 'star_distances.db'

# 3. The adjustable intensity filter.
#    This script will only include stars BRIGHTER than this magnitude.
#    A lower number means fewer, brighter stars and a much faster build time.
MAGNITUDE_LIMIT = 4.0

# ==============================================================================

def create_stars_table(db_conn, catalog_file, mag_limit):
    """
    Creates and populates the 'Stars' table.
    
    This function has been built to read raw RA/Dec strings (e.g., "hh mm ss.ss")
    from 'RAhms' and 'DEdms' columns and correctly convert them to decimal degrees 
    before calculating the unit vectors.
    """
    print("--- Step 1: Creating the 'Stars' Table (with correct conversion logic) ---")
    
    # Load the catalog using pandas
    print(f"Loading star catalog from {catalog_file}...")
    try:
        df = pd.read_csv(catalog_file)
    except FileNotFoundError:
        print(f"!!! FATAL ERROR: Cannot find the catalog file at '{catalog_file}'")
        return None
        
    # Prepare the data, looking for the raw HMS/DMS columns
    required_cols = ['HIP', 'RAhms', 'DEdms', 'Vmag']
    df.dropna(subset=required_cols, inplace=True)
    df = df[required_cols].copy()

    # Apply the magnitude filter
    print(f"Original catalog size: {len(df)} stars.")
    df_filtered = df[df['Vmag'] < mag_limit].copy()
    print(f"Filtered catalog size (Vmag < {mag_limit}): {len(df_filtered)} stars.")
    
    if len(df_filtered) == 0:
        print("!!! No stars passed the magnitude filter. Halting.")
        return None

    # --- THE NEW, CRUCIAL CONVERSION LOGIC ---
    print("Converting RA/Dec from HMS/DMS to Decimal Degrees...")
    
    # Define a robust conversion function for Right Ascension (Hours -> Degrees)
    def hms_to_deg(ra_str):
        # Handles cases where the string might not be perfect
        try:
            parts = ra_str.split()
            h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
            # The crucial formula: (h + m/60 + s/3600) * 15
            return (h + m/60 + s/3600) * 15
        except (ValueError, IndexError):
            return None # Return None for malformed strings

    # Define a robust conversion function for Declination
    def dms_to_deg(dec_str):
        # Handles cases where the string might not be perfect
        try:
            parts = dec_str.split()
            sign = -1 if parts[0].startswith('-') else 1
            d, m, s = float(parts[0]), float(parts[1]), float(parts[2])
            return (abs(d) + m/60 + s/3600) * sign
        except (ValueError, IndexError):
            return None # Return None for malformed strings

    # Apply the conversion functions to create new 'RAdeg' and 'DEdeg' columns
    df_filtered['RAdeg'] = df_filtered['RAhms'].apply(hms_to_deg)
    df_filtered['DEdeg'] = df_filtered['DEdms'].apply(dms_to_deg)
    
    # Drop any rows that failed conversion
    df_filtered.dropna(subset=['RAdeg', 'DEdeg'], inplace=True)
    # --- END OF NEW LOGIC ---

    # Convert RA/Dec degrees to 3D Cartesian unit vectors
    print("Calculating 3D Cartesian unit vectors...")
    rarad = np.deg2rad(df_filtered['RAdeg'])
    decrad = np.deg2rad(df_filtered['DEdeg'])
    df_filtered['x'] = np.cos(rarad) * np.cos(decrad)
    df_filtered['y'] = np.sin(rarad) * np.cos(decrad)
    df_filtered['z'] = np.sin(decrad)
    df_filtered['HIP'] = df_filtered['HIP'].astype(int)
    
    # Save this data to the 'Stars' table in the database
    cursor = db_conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS Stars")
    cursor.execute("""
        CREATE TABLE Stars (
            hip_id INTEGER PRIMARY KEY,
            vmag REAL NOT NULL,
            x REAL NOT NULL,
            y REAL NOT NULL,
            z REAL NOT NULL
        )
    """)
    
    # Prepare data for insertion
    stars_to_insert = df_filtered[['HIP', 'Vmag', 'x', 'y', 'z']].values
    cursor.executemany("INSERT INTO Stars (hip_id, vmag, x, y, z) VALUES (?, ?, ?, ?, ?)", stars_to_insert)
    db_conn.commit()
    
    print("'Stars' table created successfully.")
    return df_filtered

def create_distances_table(db_conn, stars_df):
    """
    Creates and populates the 'AngularDistances' table.
    This function remains unchanged as it works on the correctly converted data.
    """
    print("\n--- Step 2: Creating the 'AngularDistances' Table (This will be slow) ---")
    
    cursor = db_conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS AngularDistances")
    cursor.execute("""
        CREATE TABLE AngularDistances (
            hip1 INTEGER NOT NULL,
            hip2 INTEGER NOT NULL,
            angular_distance REAL NOT NULL,
            FOREIGN KEY(hip1) REFERENCES Stars(hip_id),
            FOREIGN KEY(hip2) REFERENCES Stars(hip_id)
        )
    """)
    
    star_ids = stars_df['HIP'].values
    vectors = stars_df[['x', 'y', 'z']].values
    
    num_stars = len(star_ids)
    total_combinations = (num_stars * (num_stars - 1)) // 2
    print(f"Calculating {total_combinations:,} unique angular distances...")

    batch_size = 100000
    batch = []
    
    for i, j in tqdm(itertools.combinations(range(num_stars), 2), total=total_combinations):
        hip1 = int(star_ids[i])
        hip2 = int(star_ids[j])
        v1 = vectors[i]
        v2 = vectors[j]
        
        dot_product = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angular_dist_deg = np.rad2deg(angle_rad)
        
        batch.append((hip1, hip2, angular_dist_deg))
        
        if len(batch) >= batch_size:
            cursor.executemany("INSERT INTO AngularDistances (hip1, hip2, angular_distance) VALUES (?, ?, ?)", batch)
            batch = []
            
    if batch:
        cursor.executemany("INSERT INTO AngularDistances (hip1, hip2, angular_distance) VALUES (?, ?, ?)", batch)

    print("All distances calculated. Now creating an index for faster lookups...")
    cursor.execute("CREATE INDEX idx_hip1 ON AngularDistances (hip1)")
    cursor.execute("CREATE INDEX idx_hip2 ON AngularDistances (hip2)")
    
    db_conn.commit()
    print("'AngularDistances' table created and indexed successfully.")


if __name__ == '__main__':
    print("--- Database Build Script Initialized ---")
    print(f"WARNING: This script can be very slow and generate a large file, especially with a high MAGNITUDE_LIMIT.")
    
    start_time = time.time()
    
    connection = sqlite3.connect(DATABASE_FILENAME)
    
    filtered_stars_df = create_stars_table(connection, CATALOG_FILENAME, MAGNITUDE_LIMIT)
    
    if filtered_stars_df is not None:
        create_distances_table(connection, filtered_stars_df)

    connection.close()
    
    end_time = time.time()
    print(f"\n--- All tasks complete. Total time: {end_time - start_time:.2f} seconds ---")
    print(f"Database saved as '{DATABASE_FILENAME}'")