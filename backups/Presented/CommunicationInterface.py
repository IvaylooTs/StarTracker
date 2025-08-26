# CommunicationInterface.py
from SystemStats import shutdown_interupt
from Flare import print_info

from IMU import *
from SystemStats import *
from WebSocket import *
from LostInSpace import lost_in_space
from FlaskServer import *
from Flare import *
import os
import sys
from LostInSpaceWrapper import *

def ActiveCalibrationSequence():
    ResetToCalibration()
    return True

def AddQuaternionAsOffsetSequence(AddedQuaternionOffset):
    AddOffsetToQuaternion(AddedQuaternionOffset)
    ResetToCalibration()

def RunLostInSpaceSequence(test=False):
    file_location = save_photo_locally()
    try:
        quaternion = None
        if not test:
            quaternion = run_lost_in_space_wrapper_sequence(file_location)   
        else:
            quaternion = lost_in_space()   
        x,y,z,w = quaternion
        AddOffsetToQuaternion((w,x,y,z))
        ActiveCalibrationSequence()
        print_info(f"Lost in space quaternion: {w,x,y,z}")
    except Exception as e:
        print_error(f"lost in space failed | {e}")
    finally:
        print_warning(f"Continue normal operation")

def RebootProgram():
    print_info("Rebooting program (gracefully)...")
    shutdown_interupt(exit_code=23)

def FullSystemRebootSequence():
    print_info("Full system reboot requested...")
    shutdown_interupt(exit_code=3)

def FullSystemShutdownSequence():
    print_info("Full system shutdown requested...")
    shutdown_interupt(exit_code=2)
