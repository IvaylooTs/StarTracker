"""
Configuration management for star tracking system.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class CameraConfig:
    """Camera intrinsic parameters"""
    image_height: int = 1964
    image_width: int = 3024
    fov_y: float = 53.0
    
    def __post_init__(self):
        # Calculate derived parameters
        self.aspect_ratio = self.image_width / self.image_height
        self.fov_x = math.degrees(
            2 * math.atan(math.tan(math.radians(self.fov_y / 2)) * self.aspect_ratio)
        )
        self.center_x = self.image_width / 2
        self.center_y = self.image_height / 2
        self.focal_length_x = (self.image_width / 2) / math.tan(math.radians(self.fov_x / 2))
        self.focal_length_y = (self.image_height / 2) / math.tan(math.radians(self.fov_y / 2))


@dataclass 
class StarTrackingConfig:
    """Configuration for star tracking algorithms"""
    # Detection parameters
    num_stars: int = 15
    
    # Matching parameters
    tolerance: float = 2.0
    min_support: int = 5
    min_matches: int = 10
    min_votes_threshold: int = 2
    
    # Algorithm parameters
    max_refinement_iterations: int = 20
    quaternion_angle_threshold: float = 1e-3
    max_search_time: float = 15.0
    
    # Numerical stability
    epsilon: float = 1e-3
    
    # File paths
    catalog_db_path: str = "../database/star_distances_sorted.db"
    catalog_hash_file: str = "../catalog/catalog_hash.pkl"