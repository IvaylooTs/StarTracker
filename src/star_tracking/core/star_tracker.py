"""
Main StarTracker class that manages state and coordinates the star tracking pipeline.
"""

import time
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

from ..utils.config import StarTrackingConfig, CameraConfig
from ..catalog.serialization import load_catalog_hash
from ..core import database as db
from ..core import coordinates as coords
from ..core.angular_distance import get_angular_distances
from ..matching.hypothesis_generation import generate_raw_votes, build_hypotheses_from_votes
from ..matching import geometric_matching as gm
from ..matching.geometric_matching import find_geometric_solutions
from ..algorithms.quest import compute_attitude_quaternion
from ..algorithms.refinement import refine_quaternion
from ..utils.scoring import load_solution_scoring
from ..core.attitude import rotational_angle_between_quaternions
from ..matching.tracking_matcher import track

@dataclass
class StarTrackingResult:
    """Result container for star tracking operations"""
    quaternion: Optional[np.ndarray] = None
    catalog_vectors: Optional[np.ndarray] = None
    star_coordinates: Optional[List[Tuple[float, float]]] = None
    solution_mapping: Optional[Dict[int, int]] = None
    processing_time: float = 0.0
    num_matched_stars: int = 0
    success: bool = False
    error_message: Optional[str] = None


class StarTracker:
    """
    Main class for star tracking and attitude determination.
    
    Manages configuration, catalog data, and provides methods for:
    - Initial star identification (lost-in-space)
    - Frame-to-frame tracking
    - Attitude refinement
    """
    
    def __init__(self, tracker_config: Optional[StarTrackingConfig] = None, camera_config: Optional[CameraConfig] = None):
        """
        Initialize StarTracker with configuration.
        Inputs
        - tracking_config: Algorithm configuration parameters
        - camera_config: Camera intrinsic parameters
        """
        
        self.config = tracker_config or StarTrackingConfig()
        self.camera = camera_config or CameraConfig()
        
        self._catalog_loaded = False
        self._catalog_hash = None
        self._catalog_angular_distances = None
        self._catalog_unit_vectors = None
        
        self.last_quaternion = None
        self.last_catalog_vectors = None
        self.last_star_coords = None
    
    def _ensure_catalog_loaded(self):
        if not self._catalog_loaded:
            try:
                print("Loading catalog data...")
                self._catalog_hash = load_catalog_hash(self.config.catalog_hash_file)
                self._catalog_angular_distances = db.load_catalog_angular_distances(self.config.catalog_db_path)
                self._catalog_unit_vectors = db.load_catalog_unit_vectors(self.config.catalog_db_path)
                self._catalog_loaded = True
                print("Catalog data loaded successfully")
            except Exception as e:
                raise RuntimeError(f"Failed to load catalog data: {e}")

    def lost_in_space(self, image_file: str, star_detection_func = None) -> StarTrackingResult:
        """
        Perform initial star identification and orientation determination
        from an image with no apriori knowlage
        Inputs:
        - image_file: Path to image file
        - star_detection_func: Optional custom star detection function
        Outputs:
        - StarTrackingResult with attitude and matching information
        """
        start_time = time.time()
        result = StarTrackingResult()
        
        try:
            self._ensure_catalog_loaded()
            
            if star_detection_func is None:
                from ..image_processing import image_processing as ip
                star_coords = ip.find_stars_with_advanced_filters(image_file, self.config.num_stars)
            else:
                star_coords = star_detection_func(image_file, self.config.num_stars)
            
            if not star_coords or len(star_coords) < self.config.min_matches:
                result.error_message = f"Insufficient stars detected: {len(star_coords)}"
                return result
            
            img_unit_vectors = coords.star_coords_to_unit_vector(
                star_coords,
                (self.camera.center_x, self.camera.center_y),
                self.camera.focal_length_x,
                self.camera.focal_length_y
            )
            
            ang_dists = get_angular_distances(
                star_coords,
                (self.camera.center_x, self.camera.center_y),
                self.camera.focal_length_x,
                self.camera.focal_length_y
            )
            
            raw_votes = generate_raw_votes(
                ang_dists, self._catalog_hash, self.config.tolerance
            )
            
            if not raw_votes:
                result.error_message = "No initial hypotheses generated"
                return result
            
            hypotheses = build_hypotheses_from_votes(
                raw_votes, self.config.min_votes_threshold
            )
            
            if not hypotheses:
                result.error_message = "No viable hypotheses after filtering"
                return result
            
            current_hypotheses = hypotheses
            solutions = []
            
            while len(current_hypotheses) >= self.config.min_matches:
                image_stars_to_try = list(current_hypotheses.keys())
                sorted_hypotheses = dict(
                    sorted(current_hypotheses.items(), key=lambda item: len(item[1]))
                )
                iteration_solutions = []
                assignment = {}
                dfs_start_time = time.time()
                
                gm.DFS(
                    assignment,
                    image_stars_to_try,
                    sorted_hypotheses,
                    ang_dists,
                    self._catalog_angular_distances,
                    self.config.tolerance,
                    iteration_solutions,
                    dfs_start_time,
                    max_time=15,
                )
                
                if iteration_solutions:
                    solutions = iteration_solutions
                    break
                else:
                    if not current_hypotheses:
                        break
                    star_to_remove = max(
                            current_hypotheses.items(), key=lambda item: len(item[1])
                    )[0]
                    print(f"Removing Star {star_to_remove} (most ambiguous) and trying again.")

                    del current_hypotheses[star_to_remove]
            
            if not solutions:
                result.error_message = "No geometrically consistent solutions found"
                return result
            
            scored_solutions = load_solution_scoring(
                solutions, ang_dists, self._catalog_angular_distances
            )
            
            best_solution, best_score = min(scored_solutions, key = lambda x: x[1])
            
            attitude_result = self._compute_attitude_from_solution(best_solution, img_unit_vectors, star_coords)
            if attitude_result is None:
                result.error_message = "Failed to compute attitude from solution"
                return result
            
            result.quaternion = attitude_result['quaternion']
            result.catalog_vectors = attitude_result['catalog_vectors']
            result.star_coordinates = attitude_result['coordinates']
            result.solution_mapping = best_solution
            result.processing_time = time.time() - start_time
            result.success = True
            
            self.last_quaternion = result.quaternion
            self.last_catalog_vectors = result.catalog_vectors
            self.last_star_coords = result.star_coordinates
            
        except Exception as e:
            result.error_message = f"Lost-in-space failed: {str(e)}"
            result.processing_time = time.time() - start_time
        
        return result

    def _compute_attitude_from_solution(self, solution_mapping: Dict[int, int], img_unit_vectors: np.ndarray, star_coords: List[Tuple[float, float]]) -> Optional[Dict[str, Any]]:
        """
        Compute attitude quaternion from a star-catalog solution mapping.
        Inputs:
        - solution_mapping: Dict mapping star indices to HIP IDs
        - img_unit_vectors: Unit vectors from image stars
        - star_coords: Pixel coordinates of stars
        Outputs:
        - Dictionary with quaternion, catalog_vectors, and coordinates
        """
        
        try:
            matched_image_vectors = []
            matched_coords = []
            for star_idx in solution_mapping.key():
                matched_image_vectors.append(img_unit_vectors[star_idx])
                matched_coords.append(star_coords[star_idx])
                
            img_matrix = np.array(matched_image_vectors)
            catalog_vectors = []
            for hip_id in solution_mapping.values():
                cat_vector = self._catalog_unit_vectors.get(hip_id)
                if cat_vector is None:
                    raise ValueError(f"404: Catalog vector for HIP {hip_id} not found")
                catalog_vectors.append(cat_vector)
            
            cat_matrix = np.array(catalog_vectors)
            quaternion = compute_attitude_quaternion(img_matrix, cat_matrix)
            
            final_quaternion = quaternion
            for i in range(self.config.max_refinement_iterations):
                refined_quaternion = refine_quaternion(final_quaternion, cat_matrix, img_matrix)
                delta_angle = rotational_angle_between_quaternions(
                    final_quaternion, refined_quaternion
                )
                if delta_angle <= self.config.quaternion_angle_threshold:
                    print(f"Quaternion refined after {i + 1} iterations")
                    break
                final_quaternion = refined_quaternion
            
            return{
                'quaternion': final_quaternion,
                'catalog_vectors': cat_matrix,
                'coordinates': matched_coords
            }
        except Exception as e:
            print(f"Error computing attitude: {e}")
            return None

    def track_frame(self, new_image_file: str, star_detection_func = None, distance_threshold: float = 10.0) -> StarTrackingResult:
        """
        Track stars from previous frame to current frame
        Inputs:
        - new_image_file: path to new image
        - star_detection_func: optional custon star detection function
        - distance_threshold; Pixel distance threshold for matching
        Outputs:
        - Updated StarTrackingResult
        """
        
        if self.last_quaternion is None:
            return self.lost_in_space(new_image_file, star_detection_func)
        
        start_time = time.time()
        result = StarTrackingResult()
        
        try:
            updated_quaternion, matches = track(
                self.last_quaternion,
                self.last_catalog_vectors,
                self.last_star_coords,
                new_image_file,
                self.camera,
                distance_threshold,
                star_detection_func
            )
            
            if updated_quaternion is not None and len(matches) >= self.config.min_matches:
                result.quaternion = updated_quaternion
                result.catalog_vectors = self.last_catalog_vectors
                result.num_matched_stars = len(matches)
                result.success = len(matches)
                
                self.last_quaternion = updated_quaternion
            else:
                print("Tracking failed, falling back to lost-in-space...")
                return self.lost_in_space(new_image_file, star_detection_func)
            
            result.processing_time = time.time() - start_time
        except Exception as e:
            result.error_message = f"Frame tracking failed: {str(e)}"
            result.processing_time = time.time() - start_time
            
        return result
    
    def get_current_attitude(self) -> Optional[np.ndarray]:
        """
        Get the current attitude quaternion
        """
        return self.last_quaternion
    
    def reset_tracking_state(self):
        """
        Reset tracking state (forces next call to use lost-in-space)
        """
        self.last_quaternion = None
        self.last_catalog_vectors = None
        self.last_star_coords = None