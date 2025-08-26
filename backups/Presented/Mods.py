from Flare import *
return_mode = "manual"

manual_source = "imu"

def get_return_mode():
    global return_mode
    return return_mode

def get_manual_source():
    global manual_source
    return manual_source

def set_return_mode(mode):
    global return_mode
    print_success(f"changed mode to {mode}")
    return_mode = mode

def set_manual_source(mode):
    global manual_source
    print_success(f"changed source to {mode}")
    manual_source = mode
