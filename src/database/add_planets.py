import sqlite3
import numpy as np

# Path to your existing database
DATABASE_FILENAME = 'src/test/star_distances_sorted.db'

# Convert RA from "hhmm" to decimal degrees
def ra_to_deg(hours, minutes):
    return (hours + minutes / 60) * 15

# Convert Dec from "±dd.d" to decimal degrees
def dec_to_deg(sign, degrees, minutes):
    decimal = degrees + minutes / 60
    return decimal if sign == '+' else -decimal

# Planet data: HIP_id, vmag, RA (h,m), Dec (sign, d, m)
planets_data = [
    (1, 1.8, (8, 28), ('+', 15, 3)),   # Mercury
    (2, -3.96, (6, 51), ('+', 21, 9)),   # Venus
    (3, 1.57, (12, 7), ('+', 0, 2)),    # Mars
    (4, -1.93, (6, 59), ('+', 22, 7)),   # Jupiter
    (5, 0.75, (0, 1), ('-', 2, 4)),      # Saturn
]

# Connect to the existing database
conn = sqlite3.connect(DATABASE_FILENAME)
cursor = conn.cursor()

# Prepare planet coordinates
planet_rows = []
for hip_id, vmag, (rah, ram), (sign, decd, decm) in planets_data:
    ra_deg = ra_to_deg(rah, ram)
    dec_deg = dec_to_deg(sign, decd, decm)
    
    rarad = np.deg2rad(ra_deg)
    decrad = np.deg2rad(dec_deg)
    
    x = np.cos(rarad) * np.cos(decrad)
    y = np.sin(rarad) * np.cos(decrad)
    z = np.sin(decrad)
    
    planet_rows.append((hip_id, vmag, x, y, z))

# Insert into Stars table
cursor.executemany(
    "INSERT OR REPLACE INTO Stars (hip_id, vmag, x, y, z) VALUES (?, ?, ?, ?, ?)",
    planet_rows
)

conn.commit()
conn.close()

print("✅ Planets added with correct RA/Dec from table.")
