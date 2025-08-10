import sqlite3

# Path to your existing database
DATABASE_FILENAME = 'digital_twin/star_distances_sorted.db'

# HIP IDs for planets to remove
planet_hip_ids = list(range(1, 9))  # 1 to 8

# Connect to the database
conn = sqlite3.connect(DATABASE_FILENAME)
cursor = conn.cursor()

# Remove planets from Stars table
cursor.execute(
    f"DELETE FROM Stars WHERE hip_id IN ({','.join('?' for _ in planet_hip_ids)})",
    planet_hip_ids
)

conn.commit()
conn.close()

print("✅ Planets removed (HIP IDs 1–8).")
