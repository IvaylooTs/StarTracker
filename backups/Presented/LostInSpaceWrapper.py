
# from QuaternionPasser import get_return_mode, get_manual_source
import time
from LostInSpace import *
from Mods import *
from IMU import *
from CommunicationInterface import *

def run_tracking():
    return (1,0,0,0)


last_timestamp_calibration = -1
last_calibration_from_lost_in_space = (0,0,0,0)

last_quaternion_from_tracking = (0,0,0,0)

def get_last_calibration_from_lost_in_space():
    global last_calibration_from_lost_in_space
    return last_calibration_from_lost_in_space

def get_last_quaternion_from_tracking():
    global last_quaternion_from_tracking
    return last_quaternion_from_tracking


def get_last_lis_calibration_timestamp():
    global last_timestamp_calibration
    return last_timestamp_calibration



start_lost_in_space = False
image_to_run = None
lis_is_running = False

ENTER_TRACKING_AFTER_LIS_IN_AUTO = True
MIN_MATCHES_TRACKING = 5
activating_tracking_mode = False # current mode 

current_lis_state = "cat"
current_system_state = "dog"
def get_current_lis_state():
    global current_lis_state
    return current_lis_state

def get_current_system_state():
    global current_system_state
    return current_system_state

def request_lost_in_space(image = None):
    global start_lost_in_space,image_to_run
    if(start_lost_in_space):
        return False
    start_lost_in_space = True
    image_to_run = image
    print_info("requested_lost_in_space activated")
    return True

def lis_wrapper_auto_loop():
    global last_quaternion_from_tracking, last_calibration_from_lost_in_space
    global last_timestamp_calibration, last_calibration_from_lost_in_space
    global image_to_run, start_lost_in_space, lis_is_running
    global current_lis_state, activating_tracking_mode

    activating_tracking_mode = False    
    last_catalog_matrix = None
    last_coords = None
    last_best_solution = None
    while True:
        time.sleep(0.4)
        current_lis_state = "Inactive"

        if(get_return_mode() == "auto"):
            # run always
            if(activating_tracking_mode is False):
                lis_is_running = True
                print_info("Entered lis in auto mode")
                file_location = save_photo_locally()
                output_package = None
                try:
                    current_lis_state = "lost in space"
                    output_package = lost_in_space(file_location)
                except:
                    print_error("No quaternion found in auto mode.")
                finally:
                    current_lis_state = "Inactive"
                
                lis_is_running = False
                if(output_package is None):
                    continue

                (quaternion, last_catalog_matrix, last_coords, last_best_solution) = output_package
                print_info(f"quaternion: {quaternion}")
                print_info(f"catalog_matrix: {last_catalog_matrix}")
                print_info(f"coords: {last_coords}")
                print_info(f"best_solution: {last_best_solution}")
                last_calibration_from_lost_in_space = quaternion
                last_timestamp_calibration = time.time()
                (x,y,z,w) = last_calibration_from_lost_in_space
                AddOffsetToQuaternion((w,x,y,z))
                ResetToCalibration()
                print_info("quaternion is found")
                activating_tracking_mode = ENTER_TRACKING_AFTER_LIS_IN_AUTO


                # last_calibration_from_lost_in_space = new_quaternion
                # last_catalog_matrix = np.array(tracking_catalog_vectors)
                # last_coords = tracking_star_coords
                # last_best_solution = tracking_solution
                # last_quaternion_from_tracking = new_quaternion
            #enter tracking

            if(activating_tracking_mode):
                print("starting save_photo_locally:")
                current_lis_state = "tracking"

                file_location = save_photo_locally()

                print("starting find_stars_with_advanced_filters:")

                detected_star_coords = ip.find_stars_with_advanced_filters(file_location, NUM_STARS)


                print("starting display_star_detections:")
                ip.display_star_detections(file_location, detected_star_coords, f"stars_identified.png")


                begin_time = time.time()
                new_quaternion, matches, tracking_catalog_vectors, tracking_star_coords, tracking_solution = track(
                    last_calibration_from_lost_in_space,
                    last_catalog_matrix,
                    last_coords,
                    last_best_solution,
                    detected_star_coords,
                    [(FOCAL_LENGTH_X, FOCAL_LENGTH_Y), (CENTER_X, CENTER_Y)],
                    MIN_MATCHES_TRACKING,
                    distance_threshold=100.0,
                )
                end_time = time.time()
        
                print(f"Tracking time: {end_time - begin_time:.3f}s")
                print(f"Tracking quaternion: {new_quaternion}")
                print(f"Matches: {matches}")
            
                if len(matches) < MIN_MATCHES_TRACKING:
                    print("Tracking failed → switching to LOST-IN-SPACE")
                    activating_tracking_mode = False
                    current_lis_state = "lost tracking"
                else:
                    print(f"Rotational angle: {rotational_angle_between_quaternions(last_calibration_from_lost_in_space, new_quaternion)}")
                    last_calibration_from_lost_in_space = new_quaternion
                    last_catalog_matrix = np.array(tracking_catalog_vectors)
                    last_coords = tracking_star_coords
                    last_best_solution = tracking_solution
                    last_calibration_from_lost_in_space = new_quaternion
                    (x,y,z,w) = last_calibration_from_lost_in_space
                    AddOffsetToQuaternion((w,x,y,z))
                    # AddOffsetToQuaternion(last_calibration_from_lost_in_space)
                    print(f"add calibration w:{w},x:{x},y:{y},z:{z}")
                    ResetToCalibration()
                    




        if(start_lost_in_space and not lis_is_running):
            print_info("we enter the algorithm")
            if(get_return_mode() != "manual"):
                print_warning("Not in manual mode. Cannot request sequence")
                start_lost_in_space = False
                continue

            print_info("starting lost in space because of request")
            lis_is_running = True

            # file_location = save_photo_locally()
            output_package= None
            try:
                current_lis_state = "lost in space"
                output_package = lost_in_space(image_to_run)
                current_lis_state = "inactive"
            except:
                print_error("No quaternion found")
            lis_is_running = False
            start_lost_in_space = False
            if(output_package is not None):
                (quaternion, catalog_matrix, coords, best_solution) = output_package
                print_info(f"quaternion: {quaternion}")
                # print_info(f"catalog_matrix: {catalog_matrix}")
                # print_info(f"coords: {coords}")
                print_info(f"best_solution: {best_solution}")
                last_timestamp_calibration = time.time()
                last_calibration_from_lost_in_space = quaternion
                print_info(f"saving last quat: {get_last_calibration_from_lost_in_space()}")
                (x,y,z,w) = last_calibration_from_lost_in_space
                AddOffsetToQuaternion((w,x,y,z))
                ResetToCalibration()
            else:
                print_error("Couldn't find quaternion")

#     while True:
#         if get_return_mode() != "auto":
#             time.sleep(0.3)
#             continue

#         quaternion = run_lost_in_space()
#         last_calibration_from_lost_in_space = quaternion
#         if(quaternion != None):
#             #calibrate IMU
#             last_timestamp_calibration = time.time()
#             quaternion = run_tracking()
#             while is_tracking_on():
#                 quaternion = run_tracking()
#                 last_quaternion_from_tracking = quaternion
#             #calibrate IMU    


def is_tracking_on():
    global activating_tracking_mode
    return activating_tracking_mode

def get_tracking_quaternion():
    return (1,0,0,0)

def is_lis_ever_called():
    return False

