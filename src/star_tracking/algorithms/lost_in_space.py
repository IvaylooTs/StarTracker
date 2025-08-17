from typing import Optional
from ..core.star_tracker import StarTracker, StarTrackingResult
from ..utils.config import StarTrackingConfig, CameraConfig

def lost_in_space(image_file: str, tracking_config: Optional[StarTrackingConfig] = None, camera_config: Optional[CameraConfig] = None, star_detection_func = None) -> StarTrackingResult:
    """
    Convenience function for lost-in-space star identification.
    This is a simplified interface that creates a StarTracker instance
    and performs star identification.
    Inputs:
    - image_file: Path to image file
    - tracking_config: Optional algorithm configuration
    - camera_config: Optional camera configuration
    - star_detection_func: Optional custom star detection function
    Outputs:
    -StarTrackingResult with attitude determination results
    """
    tracker = StarTracker(tracking_config, camera_config)
    return tracker.lost_in_space(image_file, star_detection_func)
 
def tracking_helper(image_file: str, tracking_config: Optional[StarTrackingConfig] = None, camera_config: Optional[CameraConfig] = None, star_detection_func = None) -> StarTrackingResult:
    tracker = StarTracker(tracking_config, camera_config)
    return tracker.track_frame(image_file, star_detection_func)