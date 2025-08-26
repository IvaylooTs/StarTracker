
from IMU import *
from Flare import *
from LostInSpaceWrapper import *

from Mods import *

def automatic_mode():
    quaternion = (1,0,0,0)
    trust_factor = 0


    if is_tracking_on():
        trust_factor = 2
        (w,x,y,z) =get_last_quaternion_from_tracking()
        quatenrion = (x,y,z,w) 
    else:
        quaternion = IMU_Quaternion_After_Proceessing()


    trust_factor |= is_IMU_calibrated_recently()

    return quaternion, trust_factor


# from specific manual target, set in manual mode
def get_manually_quaterion():
    quaternion = (1,0,0,0)
    trust_factor = 0


    match get_manual_source():
        # case "tracking":
        #     quaternion = get_last_quaternion_from_tracking()
        #     trust_factor = 2 * is_tracking_on()
        case "lostinspace":
            (w,x,y,z) = get_last_calibration_from_lost_in_space()
            quaternion = (y,x,x,z)
            #               z w x y
            # quaternion = (y,z,w,x)
            quaternion = (z,w,x,y)
            trust_factor = 0
        case "imu":
            quaternion = IMU_Quaternion_After_Proceessing()
            trust_factor |= is_IMU_calibrated_recently()
        case _:
            quaternion = (1,0,0,0)
            trust_factor = -1
            print_error("Invalid mode selected, returning identity quaterinon")
    
    return quaternion, trust_factor


def passed_Quaternion():
    final_quaternion = (1,0,0,0)
    trust_factor = 0
    if get_return_mode() == "auto":
        final_quaternion, trust_factor = automatic_mode()
    elif get_return_mode() == "manual":
        final_quaternion, trust_factor = get_manually_quaterion()

    return final_quaternion, trust_factor