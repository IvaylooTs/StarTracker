import sqlite3
import math

# Define the path to your database file
DATABASE_FILENAME = 'digital_twin/star_distances_sorted.db'

def angular_distance_cartesian(x1, y1, z1, x2, y2, z2):
    """
    Computes the angular distance in degrees between two points
    given in 3D Cartesian coordinates.
    """
    # Calculate the dot product of the two vectors
    dot_product = x1 * x2 + y1 * y2 + z1 * z2

    # Calculate the magnitude (length) of each vector
    mag1 = math.sqrt(x1**2 + y1**2 + z1**2)
    mag2 = math.sqrt(x2**2 + y2**2 + z2**2)

    # To avoid division by zero if a star is at the origin (0,0,0)
    if mag1 == 0 or mag2 == 0:
        return 0.0

    # Calculate the cosine of the angle using the dot product formula. [7]
    # Clamp the value between -1.0 and 1.0 to prevent potential floating-point errors.
    cos_angle = max(-1.0, min(1.0, dot_product / (mag1 * mag2)))

    # Calculate the angle in radians and convert to degrees
    angle_rad = math.acos(cos_angle)
    return math.degrees(angle_rad)

# Connect to the SQLite database
conn = sqlite3.connect(DATABASE_FILENAME)
cursor = conn.cursor()

# Get all "planets" (as defined by HIP IDs 1–8) with their Cartesian coordinates
cursor.execute("SELECT hip_id, x, y, z FROM Stars WHERE hip_id BETWEEN 1 AND 8")
planets = cursor.fetchall()

# Get all other stars with their Cartesian coordinates
cursor.execute("SELECT hip_id, x, y, z FROM Stars WHERE hip_id NOT BETWEEN 1 AND 8")
stars = cursor.fetchall()

# A list to hold all the data to be inserted
distances_to_insert = []

# Calculate angular distances (planet ↔ star)
for planet_id, x_p, y_p, z_p in planets:
    for star_id, x_s, y_s, z_s in stars:
        # Use the new function designed for Cartesian coordinates
        dist = angular_distance_cartesian(x_p, y_p, z_p, x_s, y_s, z_s)
        
        # Add the data to our list for bulk insertion
        distances_to_insert.append((planet_id, star_id, dist))
        # The original code inserted the reverse pair as well, so we maintain that logic
        distances_to_insert.append((star_id, planet_id, dist))

# Insert all the calculated distances into the correct table `AngularDistances`
# Using executemany is more efficient than executing in a loop
cursor.executemany(
    "INSERT OR REPLACE INTO AngularDistances (hip1, hip2, angular_distance) VALUES (?, ?, ?)",
    distances_to_insert
)

# Commit the changes to the database and close the connection
conn.commit()
conn.close()

print("✅ Angular distances between planets and stars added.")