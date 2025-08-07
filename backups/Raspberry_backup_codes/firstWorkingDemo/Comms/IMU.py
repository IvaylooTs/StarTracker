# imu.py
import math
from smbus2 import SMBus

import time
import json
from Flare import *
from Math import *
# ============================================================
#                       BNO055 Sensor Configuration 
# ============================================================

MAX_REBOOT_ATTEMPTS = 2
REBOOT_DELAY = 1 # seconds

IMU_I2C_ADDR = 0x28
BUS = 1

OPR_MODE_ADDR = 0x3D
CALIBRATION_REG_START = 0x55
CALIBRATION_REG_LEN = 22

# modes
CONFIG_MODE = 0x00
NDOF_FMC_OFF_MODE = 0x0C

# Euler angles registers (from datasheet, little endian, 16-bit signed)
# Each angle is in units of 1/16 degrees
EULER_H_LSB = 0x1A
EULER_R_LSB = 0x1C
EULER_P_LSB = 0x1E

SYS_STATUS_ADDR = 0x39
SYS_ERR_ADDR = 0x3A

CHIP_ID_ADDR = 0x00
EXPECTED_CHIP_ID = 0xA0

bus = SMBus(BUS)

IMU_AVAILABLE = False
def is_IMU_available():
    global IMU_AVAILABLE
    return IMU_AVAILABLE
# ============================================================
#                       setup functions 
# ============================================================

def set_mode(mode):
    bus.write_byte_data(IMU_I2C_ADDR, OPR_MODE_ADDR, CONFIG_MODE)
    time.sleep(0.05)
    bus.write_byte_data(IMU_I2C_ADDR, OPR_MODE_ADDR, mode)
    time.sleep(0.05)

def write_calibration(data):
    assert len(data) == 22
    bus.write_i2c_block_data(IMU_I2C_ADDR, CALIBRATION_REG_START, data)
    print_warning("Calibration written to sensor.")

def load_calibration(filename="bno055_cal.json"):
    with open(filename, "r") as f:
        return json.load(f)


# ===========================================================
#                       Euler Angles Reading
# ===========================================================

def read_euler():
    if not is_IMU_available():
        return 0,0,0
    data = bus.read_i2c_block_data(IMU_I2C_ADDR, EULER_H_LSB, 6)  # 2 bytes each for heading, roll, pitch
    # convert from little endian signed 16-bit
    heading = int.from_bytes(data[0:2], byteorder='little', signed=True) / 16.0
    roll = int.from_bytes(data[2:4], byteorder='little', signed=True) / 16.0
    pitch = int.from_bytes(data[4:6], byteorder='little', signed=True) / 16.0
    return heading, roll, pitch

# ===========================================================
#                     Read quaternions
# ===========================================================

InvertedQuaternion = (1, 0, 0, 0)  # Identity quaternion for inversion

AddedQuaternionOffset = (1, 0, 0, 0)  # Offset to be added to the quaternion after calibration

def AddOffsetToQuaternion(quaternion):
    global AddedQuaternionOffset
    AddedQuaternionOffset = quaternion

# used by Jitter detection
LastQuaternion =(1, 0, 0, 0) # used for gitter cleanup
InCalibration = False
bugDetected = False

def differenceInQuaternions(q1, q2):
    return math.sqrt((q1[0] - q2[0])**2 + 
                         (q1[1] - q2[1])**2 + 
                         (q1[2] - q2[2])**2 + 
                         (q1[3] - q2[3])**2)


def DetectJitter(quaternion, threshold=0.25):
    global LastQuaternion,bugDetected
    if(InCalibration == True):
        LastQuaternion = quaternion
        return quaternion  # Skip jitter detection during calibration
    if LastQuaternion == (1, 0, 0, 0):
        LastQuaternion = quaternion
        return quaternion
    else:
        diff = differenceInQuaternions(LastQuaternion, quaternion)
        if diff > threshold:  # threshold for jitter detection
            print_warning(f"Jitter detected Difference: {diff}")
            if(bugDetected == True):
                time.sleep(0.1)
                print_info("Last quaternion glitch wasn't a glitch returning to save state.")
                bugDetected = False
                LastQuaternion = quaternion
                return quaternion
            else:
                bugDetected = True
                if(differenceInQuaternions(quaternion, Quaternion_After_Proceessing()) <= threshold):
                    LastQuaternion = quaternion
                    return LastQuaternion
                else:
                    return LastQuaternion
                return LastQuaternion
        else:
            bugDetected = False
            LastQuaternion = quaternion
            return quaternion


def Quaternion_After_Proceessing():
    if not is_IMU_available():
        return (0,0,0,0)
    # Read the quaternion from the sensor
    w, x, y, z = read_quaternion()
    # Normalize the quaternion
    norm = math.sqrt(w**2 + x**2 + y**2 + z**2)
    while norm == 0:
        w, x, y, z = read_quaternion()
        norm = math.sqrt(w**2 + x**2 + y**2 + z**2)

    w /= norm
    x /= norm
    y /= norm
    z /= norm
    
    # print("invert quaternion:", InvertedQuaternion)
    # print("quaternion:", w, x, y, z)

    new_quaternion = multiply_quaternions(InvertedQuaternion, (w, x, y, z))
    # print("new quaternion:", new_quaternion)
    
    fixed_quaternion = DetectJitter(new_quaternion)
    # print(AddedQuaternionOffset)
    return multiply_quaternions(AddedQuaternionOffset,fixed_quaternion)

def read_quaternion():
    if not is_IMU_available():
        return (0,0,0,0)
    # Quaternion registers (0x20 - 0x27), little-endian, 16-bit signed integers
    QUATERNION_REG_START = 0x20
    data = bus.read_i2c_block_data(IMU_I2C_ADDR, QUATERNION_REG_START, 8)

    # Convert bytes to 16-bit signed integers
    w = int.from_bytes(data[0:2], byteorder='little', signed=True) / (1 << 14)
    x = int.from_bytes(data[2:4], byteorder='little', signed=True) / (1 << 14)
    y = int.from_bytes(data[4:6], byteorder='little', signed=True) / (1 << 14)
    z = int.from_bytes(data[6:8], byteorder='little', signed=True) / (1 << 14)

    return w, x, y, z

# ===========================================================
#                    Restart calibration
# ===========================================================

def StartCalibration():
    global InvertedQuaternion
    global InCalibration
    global LastQuaternion
    InCalibration = True
    print("Starting software calibration of IMU")
    
    q = read_quaternion()
    while not is_normalized(q[0], q[1], q[2], q[3]):
        print("Quaternion not normalized, retrying...")
        time.sleep(0.1)
        q = read_quaternion()
        print(f"Quaternion: w={q[0]:.8f}, x={q[1]:.8f}, y={q[2]:.8f}, z={q[3]:.8f}");
        current_norm = math.sqrt(q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)
        print("current norm", current_norm)

    print(f"Quaternion: w={q[0]:.8f}, x={q[1]:.8f}, y={q[2]:.8f}, z={q[3]:.8f}")
    
    InvertedQuaternion = inverse_quaternion(q[0], q[1], q[2], q[3])
    print(f"Inverted Quaternion: w={InvertedQuaternion[0]:.8f}, x={InvertedQuaternion[1]:.8f}, y={InvertedQuaternion[2]:.8f}, z={InvertedQuaternion[3]:.8f}")
    InCalibration = False
    LastQuaternion = (1,0,0,0)

def check_imu_connection():
    try:
        chip_id = bus.read_byte_data(IMU_I2C_ADDR, CHIP_ID_ADDR)
        if chip_id == EXPECTED_CHIP_ID:
            print_info("BNO055 IMU detected and responding.")
            return True
        else:
            print_warning(f"Unexpected CHIP ID: 0x{chip_id:X}")
            return False
    except OSError as e:
        print_error("BNO055 IMU not detected on I2C bus.")
        return False

def test_IMU():
    # if not is_IMU_available():
    #     print_warning("IMU not available — skipping IMU tests.")
    #     return False

    try:
        # Read self-test results
        SELFTEST_RESULT_ADDR = 0x36
        result = bus.read_byte_data(IMU_I2C_ADDR, SELFTEST_RESULT_ADDR)

        accel_ok = (result >> 0) & 0x01
        mag_ok   = (result >> 1) & 0x01
        gyro_ok  = (result >> 2) & 0x01
        mcu_ok   = (result >> 3) & 0x01

        print_info("Self-Test Results:")
        print_info(f" - Accelerometer: {'PASS' if accel_ok else 'FAIL'}")
        print_info(f" - Magnetometer:  {'PASS' if mag_ok else 'FAIL'}")
        print_info(f" - Gyroscope:     {'PASS' if gyro_ok else 'FAIL'}")
        print_info(f" - MCU:           {'PASS' if mcu_ok else 'FAIL'}")

        # System Status
        time.sleep(0.1)
        sys_status = bus.read_byte_data(IMU_I2C_ADDR, SYS_STATUS_ADDR)
        sys_err = bus.read_byte_data(IMU_I2C_ADDR, SYS_ERR_ADDR)

        if sys_status == 5:
            print_info("System status: Fully operational.")
        elif sys_status == 0:
            print_warning("System idle.")
        elif sys_status == 1:
            print_error(f"System error! Error code: {sys_err}")
        else:
            print_warning(f"System status code: {sys_status}")

        if accel_ok and mag_ok and gyro_ok and mcu_ok and (sys_status == 5 or sys_status == 0 or sys_status == 1 or sys_status == 133):
            print_info("All BNO055 self-tests passed.")
            return True
        else:
            print_warning("One or more self-tests failed.")
            return (accel_ok, mag_ok, gyro_ok, mcu_ok, sys_status)


    except Exception as e:
        print_error(f"Exception during BNO055 test: {e}")
        return False

def InitIMU():
    global IMU_AVAILABLE

    # This is used to change modes. Can be:
    # True - everything is fine
    # (a,g,m,s) - accelerometer, gyroscope, magentometer, system failed 
    sys_info = None
    print_header("Starting BNO055 Calibration Process")
    for i in range(MAX_REBOOT_ATTEMPTS):
        print_info(f"System calibration attempt: {i+1}/{MAX_REBOOT_ATTEMPTS}")
        if(check_imu_connection() == False):
            print_fatal_error("BNO055 IMU is unresponsive.")
            print_warning("retrying initialization again...")
            time.sleep(REBOOT_DELAY)
            continue
        
        IMU_AVAILABLE = True
        print_info("Setting CONFIG mode")
        set_mode(CONFIG_MODE)
        
        print_info("Running tests:")
        sys_info = test_IMU()
        if sys_info == True:
            print_success("All IMU tests passed!")
        else:
            print_error("Front testing failed!")
            print_warning("retrying initialization again...")
            time.sleep(REBOOT_DELAY)
            continue

        print_info("loading calibration data...")
        cal = load_calibration()
        write_calibration(cal)

        print_info("Entering NDOF_FMC_OFF mode to lock calibration...")
        set_mode(NDOF_FMC_OFF_MODE)
        time.sleep(0.20)
        print_warning("Final testing...")
        
        sys_info = test_IMU()
        if sys_info == True:
            print_success("All IMU tests passed!")
        else:
            print_error("Final testing failed!")
            print_warning("retrying initialization again...")
            time.sleep(REBOOT_DELAY)
            continue
            

        print_success("IMU setup successful!")

        print_info("Format - Euler angles - (Heading, Roll, Pitch)")
        print_info("Format - Quaternion - (w, x, y, z)")
        
        return True

    print_error("IMU module failed")
    return False