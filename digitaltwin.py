import math
import itertools
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, ICRS
from astroquery.gaia import Gaia
from scipy.spatial.transform import Rotation as R # For converting rotation matrix to quaternion/Euler angles

# --- 1. Star Class ---

class Star:
    """Represents a star with celestial coordinates (RA, Dec) and a unique ID."""
    def __init__(self, star_id, ra_deg, dec_deg, magnitude=None):
        self.id = star_id
        self.ra = math.radians(ra_deg)  # Right Ascension in radians
        self.dec = math.radians(dec_deg) # Declination in radians
        self.magnitude = magnitude

    def to_cartesian(self):
        """Converts spherical coordinates (RA, Dec) to 3D Cartesian coordinates on a unit sphere."""
        x = math.cos(self.dec) * math.cos(self.ra)
        y = math.cos(self.dec) * math.sin(self.ra)
        z = math.sin(self.dec)
        return np.array([x, y, z])

    def __repr__(self):
        return f"Star(ID={self.id}, RA={math.degrees(self.ra):.2f}°, Dec={math.degrees(self.dec):.2f}°)"

class StarCatalog:
    """A simplified star catalog."""
    def __init__(self, stars):
        self.stars = {star.id: star for star in stars}
        self.star_list = list(stars) # For easier iteration

    def get_star(self, star_id):
        return self.stars.get(star_id)

    def get_all_stars(self):
        return self.star_list

# --- 2. Star Tracker Class ---

class StarTracker:
    """
    A simplified star tracker using a triangle-based pattern recognition method
    and SVD-based attitude determination.
    """
    def __init__(self, catalog, camera_fov_deg=20.0, image_width_pixels=1024, image_height_pixels=1024):
        self.catalog = catalog
        self.camera_fov_rad = math.radians(camera_fov_deg)
        self.image_width_pixels = image_width_pixels
        self.image_height_pixels = image_height_pixels
        
        # Simplified focal length calculation (assuming square pixels and pinhole camera model)
        # f_pixels = image_width_pixels / (2 * tan(FOV_rad / 2))
        self.focal_length_pixels = self.image_width_pixels / (2 * math.tan(self.camera_fov_rad / 2))

        self.catalog_triplets = self._precompute_catalog_triplets()

    def _angular_distance_between_stars(self, star1, star2):
        """Calculates the angular distance (in radians) between two stars."""
        vec1 = star1.to_cartesian()
        vec2 = star2.to_cartesian()
        dot_product = np.dot(vec1, vec2)
        dot_product = np.clip(dot_product, -1.0, 1.0) # Clip to avoid floating point errors
        return math.acos(dot_product)

    def _pixel_distance_to_angular_distance(self, pixel_dist):
        """
        Converts pixel distance to approximate angular distance (in radians).
        This is a highly simplified linear approximation, not a true projection.
        Used only for triplet matching, not for attitude determination.
        """
        return pixel_dist / self.focal_length_pixels # More accurate with focal length

    def _precompute_catalog_triplets(self):
        """
        Precomputes all unique triplets of stars from the catalog and their
        sorted angular distances. This forms the "hash table" for matching.
        """
        print("Precomputing catalog triplets...")
        triplets_data = {}
        if len(self.catalog.get_all_stars()) < 3:
            print("Warning: Not enough stars in catalog to precompute triplets.")
            return {}

        for s1, s2, s3 in itertools.combinations(self.catalog.get_all_stars(), 3):
            d12 = self._angular_distance_between_stars(s1, s2)
            d13 = self._angular_distance_between_stars(s1, s3)
            d23 = self._angular_distance_between_stars(s2, s3)
            sorted_distances = tuple(sorted([d12, d13, d23]))
            if sorted_distances not in triplets_data:
                triplets_data[sorted_distances] = []
            triplets_data[sorted_distances].append((s1.id, s2.id, s3.id))
        print(f"Precomputed {len(triplets_data)} unique catalog triplets.")
        return triplets_data

    def detect_stars_in_image(self, image_data):
        """
        Simulates star detection in an image.
        `image_data` is expected to be a list of (x, y) pixel coordinates.
        The origin (0,0) is assumed to be the top-left corner of the image.
        """
        detected_stars = []
        for i, (x, y) in enumerate(image_data):
            detected_stars.append({'id': f"IMG_{i}", 'pixel_coords': (x, y)})
        print(f"Detected {len(detected_stars)} stars in the image.")
        return detected_stars

    def _calculate_image_triplet_distances(self, img_star1, img_star2, img_star3):
        """
        Calculates pixel distances for an image triplet and converts to angular.
        """
        p1 = np.array(img_star1['pixel_coords'])
        p2 = np.array(img_star2['pixel_coords'])
        p3 = np.array(img_star3['pixel_coords'])

        d12_pixel = np.linalg.norm(p1 - p2)
        d13_pixel = np.linalg.norm(p1 - p3)
        d23_pixel = np.linalg.norm(p2 - p3)

        d12_angular = self._pixel_distance_to_angular_distance(d12_pixel)
        d13_angular = self._pixel_distance_to_angular_distance(d13_pixel)
        d23_angular = self._pixel_distance_to_angular_distance(d23_pixel)

        return tuple(sorted([d12_angular, d13_angular, d23_angular]))

    def identify_stars(self, detected_image_stars, tolerance_rad=0.001):
        """
        Identifies stars in the image by matching triplets to the catalog.
        Returns a dictionary mapping image star IDs to catalog star IDs.
        """
        if len(detected_image_stars) < 3:
            print("Not enough stars detected for triplet matching.")
            return {}

        image_to_catalog_votes = {} # {image_star_id: {catalog_star_id: vote_count}}

        for img_s1, img_s2, img_s3 in itertools.combinations(detected_image_stars, 3):
            img_distances = self._calculate_image_triplet_distances(img_s1, img_s2, img_s3)

            for catalog_dist_tuple, catalog_star_ids_list in self.catalog_triplets.items():
                if all(abs(img_d - cat_d) < tolerance_rad for img_d, cat_d in zip(img_distances, catalog_dist_tuple)):
                    for catalog_triplet_ids in catalog_star_ids_list:
                        for img_s_obj in [img_s1, img_s2, img_s3]:
                            img_id = img_s_obj['id']
                            if img_id not in image_to_catalog_votes:
                                image_to_catalog_votes[img_id] = {}
                            for cat_id in catalog_triplet_ids:
                                image_to_catalog_votes[img_id][cat_id] = \
                                    image_to_catalog_votes[img_id].get(cat_id, 0) + 1

        final_matches = {}
        for img_id, votes in image_to_catalog_votes.items():
            if votes:
                best_match_cat_id = max(votes, key=votes.get)
                final_matches[img_id] = best_match_cat_id
        
        return final_matches

    def _pixels_to_camera_vectors(self, pixel_coords):
        """
        Converts pixel coordinates to normalized 3D vectors in the camera frame.
        Assumes origin (0,0) is top-left, camera boresight is +Z, +X is right, +Y is down.
        The center of the image is (image_width_pixels/2, image_height_pixels/2).
        """
        cx = self.image_width_pixels / 2
        cy = self.image_height_pixels / 2

        # Convert pixel coordinates to coordinates relative to the image center
        x_prime = pixel_coords[0] - cx
        y_prime = pixel_coords[1] - cy

        # Convert to normalized camera coordinates
        # For a pinhole camera, x_cam = x_prime / f, y_cam = y_prime / f, z_cam = 1
        # Then normalize the vector.
        # Note: y_prime is inverted because pixel Y increases downwards, but camera Y usually points up.
        # This depends on camera mounting. For simplicity, let's assume +Y is down in camera frame.
        
        # The vector in camera coordinates is (x_prime, y_prime, focal_length_pixels)
        # We need to normalize this vector.
        vector = np.array([x_prime, y_prime, self.focal_length_pixels])
        return vector / np.linalg.norm(vector)

    def determine_attitude(self, matched_stars, detected_image_stars_context):
        """
        Determines the camera's orientation (attitude) using identified stars.
        This implements a solution to Wahba's problem using SVD.
        
        Args:
            matched_stars (dict): Dictionary mapping image star IDs to catalog star IDs.
            detected_image_stars_context (list): List of dictionaries with 'id' and 'pixel_coords'
                                                 for all detected image stars.
        
        Returns:
            tuple: (rotation_matrix, quaternion_xyzw, euler_angles_deg) or None if not enough matches.
        """
        if len(matched_stars) < 2: # Need at least 2 non-collinear stars for 3D attitude, 3 for robustness
            print("\nNot enough identified stars to determine attitude (need at least 2).")
            return None

        # Prepare observed (camera frame) and reference (celestial frame) vectors
        observed_vectors = [] # b_i
        reference_vectors = [] # r_i

        print("\n--- Preparing Vectors for Attitude Determination ---")
        for img_id, cat_id in matched_stars.items():
            img_star_data = next((s for s in detected_image_stars_context if s['id'] == img_id), None)
            cat_star_data = self.catalog.get_star(cat_id)

            if img_star_data and cat_star_data:
                # Convert pixel coordinates to camera frame unit vectors
                b_i = self._pixels_to_camera_vectors(img_star_data['pixel_coords'])
                observed_vectors.append(b_i)

                # Get celestial frame unit vectors from catalog
                r_i = cat_star_data.to_cartesian()
                reference_vectors.append(r_i)
                
                print(f"  Image Star '{img_id}' (px={img_star_data['pixel_coords']}) -> Camera Vec: {b_i.round(3)}")
                print(f"  Catalog Star '{cat_id}' (RA/Dec={math.degrees(cat_star_data.ra):.2f}°, {math.degrees(cat_star_data.dec):.2f}°) -> Ref Vec: {r_i.round(3)}")
            else:
                print(f"  Warning: Data missing for match: Image '{img_id}' or Catalog '{cat_id}'. Skipping.")

        if len(observed_vectors) < 2:
            print("Not enough valid vector pairs to determine attitude.")
            return None

        observed_vectors = np.array(observed_vectors).T # Shape (3, N)
        reference_vectors = np.array(reference_vectors).T # Shape (3, N)

        # --- Solve Wahba's Problem using SVD ---
        # B = sum(w_i * r_i * b_i^T) -- Here, weights w_i are assumed to be 1
        B = reference_vectors @ observed_vectors.T

        U, S, Vt = np.linalg.svd(B)
        V = Vt.T

        # Ensure a right-handed coordinate system (no reflection)
        # If det(V @ U.T) is -1, it means there's a reflection.
        # We correct this by flipping the sign of the last column of V.
        d = np.diag([1, 1, np.linalg.det(V @ U.T)])
        
        # The optimal rotation matrix R that transforms vectors from camera frame to inertial frame
        # r_i = R @ b_i
        rotation_matrix = V @ d @ U.T

        print("\n--- Attitude Determination Results ---")
        print("Calculated Rotation Matrix (Camera to Inertial Frame):")
        print(rotation_matrix.round(4))

        # Convert rotation matrix to quaternion (scalar-last, x,y,z,w)
        rotation = R.from_matrix(rotation_matrix)
        quaternion_xyzw = rotation.as_quat() # (x, y, z, w)
        euler_angles_deg = rotation.as_euler('zyx', degrees=True) # Yaw, Pitch, Roll in degrees

        print(f"\nQuaternion (x, y, z, w): {quaternion_xyzw.round(4)}")
        print(f"Euler Angles (Z-Y-X, degrees - Yaw, Pitch, Roll): {euler_angles_deg.round(2)}")

        return rotation_matrix, quaternion_xyzw, euler_angles_deg

# --- Function to fetch stars from Gaia DR3 (ICRS) ---

def fetch_stars_from_gaia(ra_center_deg, dec_center_deg, radius_deg, magnitude_limit=6.0, max_stars=100):
    """
    Fetches stars from the Gaia DR3 catalog using astroquery.
    Queries a circular region around a given RA/Dec.
    """
    print(f"\nAttempting to fetch stars from Gaia DR3 (ICRS) around RA={ra_center_deg}°, Dec={dec_center_deg}° with radius={radius_deg}°...")
    try:
        # Set the center of the query region
        center_coord = SkyCoord(ra=ra_center_deg * u.deg, dec=dec_center_deg * u.deg, frame='icrs')
        
        # Query Gaia DR3 for sources within the specified radius and magnitude limit
        # Limiting to brighter stars (G_mag < magnitude_limit) and a reasonable number
        # to keep the query manageable for a demo.
        # 'phot_g_mean_mag' is the G-band mean magnitude in Gaia DR3
        job = Gaia.cone_search_async(
            center_coord, 
            radius=radius_deg * u.deg, 
            # ADQL query to filter by magnitude and limit results
            # Note: The 'TOP' clause is database-specific, 'limit' is generally preferred
            # but cone_search_async doesn't directly support it in its parameters.
            # We'll filter after fetching if needed.
            # For a more precise query, one might use Gaia.launch_job_async with ADQL.
        )
        table = job.get_results()

        # Filter by magnitude and limit the number of stars if the initial query returns too many
        filtered_stars = []
        for row in table:
            if row['phot_g_mean_mag'] < magnitude_limit:
                filtered_stars.append(row)
            if len(filtered_stars) >= max_stars:
                break
        
        stars = []
        for i, row in enumerate(filtered_stars):
            # Gaia source_id is a good unique ID
            star_id = row['source_id']
            ra = row['ra']
            dec = row['dec']
            magnitude = row['phot_g_mean_mag']
            stars.append(Star(star_id, ra, dec, magnitude))
        
        print(f"Successfully fetched {len(stars)} stars from Gaia DR3.")
        return stars

    except (TimeoutError, NoResultFound, Exception) as e:
        print(f"Failed to fetch stars from Gaia DR3 API: {e}")
        print("Falling back to mock star catalog.")
        return None

# --- Example Usage ---

if __name__ == "__main__":
    # --- Configuration for fetching real stars ---
    # Choose a region in the sky for the catalog.
    # For demonstration, a small region in Ursa Major (near the mock stars)
    central_ra = 20.8 # degrees
    central_dec = 45.2 # degrees
    search_radius = 2.0 # degrees (This should be larger than camera FOV to ensure stars are found)
    bright_magnitude_limit = 7.0 # Only fetch stars brighter than this magnitude (lower number = brighter)
    max_fetched_stars = 50 # Limit the number of stars fetched for performance in demo

    # Attempt to fetch real stars from Gaia DR3
    real_stars_data = fetch_stars_from_gaia(
        central_ra, central_dec, search_radius, bright_magnitude_limit, max_fetched_stars
    )

    if real_stars_data:
        catalog = StarCatalog(real_stars_data)
    else:
        # Fallback to mock star catalog if API call fails
        mock_stars = [
            Star(1, 20.62, 45.28, 2.0),
            Star(2, 20.68, 45.45, 2.2),
            Star(3, 20.80, 45.10, 3.0),
            Star(4, 21.00, 44.90, 2.5),
            Star(5, 20.50, 45.00, 3.5),
            Star(6, 21.10, 45.30, 2.8),
            Star(7, 20.75, 45.05, 3.1),
        ]
        catalog = StarCatalog(mock_stars)
        print("Using mock star catalog.")

    # Initialize the star tracker
    # Assume a camera with 20-degree FOV and 1024x1024 pixel resolution
    star_tracker = StarTracker(catalog, camera_fov_deg=20.0, image_width_pixels=1024, image_height_pixels=1024)

    # --- Simulate a captured sky image ---
    # These pixel coordinates represent detected stars in a 1024x1024 image.
    # They are slightly perturbed from ideal positions to simulate noise/error.
    # For this demo, these mock pixel positions are *not* dynamically generated
    # from the fetched Gaia stars, but are kept consistent with the original mock
    # data's relative positions. In a real application, these would come from
    # actual image processing of a captured sky image.

    detected_image_stars_pixels = [
        (503, 516), # Corresponds to a star near RA 20.62, Dec 45.28
        (506, 525), # Corresponds to a star near RA 20.68, Dec 45.45
        (510, 508), # Corresponds to a star near RA 20.80, Dec 45.10
        (520, 495), # Corresponds to a star near RA 21.00, Dec 44.90
        (498, 500), # Corresponds to a star near RA 20.50, Dec 45.00
        (525, 518), # Corresponds to a star near RA 21.10, Dec 45.30
        (508, 510), # Corresponds to a star near RA 20.75, Dec 45.05
        (600, 600), # A "false" star or noise
    ]

    detected_stars = star_tracker.detect_stars_in_image(detected_image_stars_pixels)

    # --- Identify the stars ---
    # Tolerance for matching angular distances (in radians)
    # 0.005 radians is approx 0.28 degrees
    identified_matches = star_tracker.identify_stars(detected_stars, tolerance_rad=0.005)

    print("\n--- Identified Star Matches ---")
    if identified_matches:
        for img_id, cat_id in identified_matches.items():
            print(f"Image Star '{img_id}' matched to Catalog Star '{cat_id}'")
    else:
        print("No matches found.")

    # --- Determine Satellite Orientation ---
    # Pass the detected_stars list for context in the conceptual explanation
    attitude_results = star_tracker.determine_attitude(identified_matches, detected_stars)

    if attitude_results:
        rotation_matrix, quaternion_xyzw, euler_angles_deg = attitude_results
        print("\nSuccessfully determined camera orientation!")
        # You can now use these values for your satellite's attitude control system.
    else:
        print("\nCould not determine camera orientation.")