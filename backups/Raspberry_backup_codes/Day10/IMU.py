# imu.py
import math
from smbus2 import SMBus

import time
import json
from Flare import print_header
from Math import *
# ============================================================
#                       BNO055 Sensor Configuration 
# ============================================================

ADDR = 0x28
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

bus = SMBus(BUS)

def set_mode(mode):
    bus.write_byte_data(ADDR, OPR_MODE_ADDR, CONFIG_MODE)
    time.sleep(0.05)
    bus.write_byte_data(ADDR, OPR_MODE_ADDR, mode)
    time.sleep(0.05)

def write_calibration(data):
    assert len(data) == 22
    bus.write_i2c_block_data(ADDR, CALIBRATION_REG_START, data)
    print("Calibration written to sensor.")

def load_calibration(filename="bno055_cal.json"):
    with open(filename, "r") as f:
        return json.load(f)


# ===========================================================
#                       Euler Angles Reading
# ===========================================================

def read_euler():
    data = bus.read_i2c_block_data(ADDR, EULER_H_LSB, 6)  # 2 bytes each for heading, roll, pitch
    # convert from little endian signed 16-bit
    heading = int.from_bytes(data[0:2], byteorder='little', signed=True) / 16.0
    roll = int.from_bytes(data[2:4], byteorder='little', signed=True) / 16.0
    pitch = int.from_bytes(data[4:6], byteorder='little', signed=True) / 16.0
    return heading, roll, pitch


InvertedQuaternion = (1, 0, 0, 0)  # Identity quaternion for inversion

LastQuaternion =(1, 0, 0, 0) # used for gitter cleanup

InCalibration = False

def DetectJitter(quaternion, threshold=0.25):
    global LastQuaternion
    if(InCalibration == True):
        LastQuaternion = quaternion
        return quaternion  # Skip jitter detection during calibration
    if LastQuaternion == (1, 0, 0, 0):
        LastQuaternion = quaternion
        return quaternion
    else:
        diff = math.sqrt((LastQuaternion[0] - quaternion[0])**2 + 
                         (LastQuaternion[1] - quaternion[1])**2 + 
                         (LastQuaternion[2] - quaternion[2])**2 + 
                         (LastQuaternion[3] - quaternion[3])**2)
        if diff > threshold:  # threshold for jitter detection
            print("Jitter detected, returning last quaternion. Difference:", diff)
            return LastQuaternion
        else:
            LastQuaternion = quaternion
            return quaternion


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

def QuaternionAfterProceessing():
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
    
    return DetectJitter(new_quaternion)

# ===========================================================
#                       quaternion math
# ===========================================================

def read_quaternion():
    # Quaternion registers (0x20 - 0x27), little-endian, 16-bit signed integers
    QUATERNION_REG_START = 0x20
    data = bus.read_i2c_block_data(ADDR, QUATERNION_REG_START, 8)

    # Convert bytes to 16-bit signed integers
    w = int.from_bytes(data[0:2], byteorder='little', signed=True) / (1 << 14)
    x = int.from_bytes(data[2:4], byteorder='little', signed=True) / (1 << 14)
    y = int.from_bytes(data[4:6], byteorder='little', signed=True) / (1 << 14)
    z = int.from_bytes(data[6:8], byteorder='little', signed=True) / (1 << 14)

    return w, x, y, z

# ===========================================================
#                    Start calibration process
# ===========================================================



print_header("Starting BNO055 Calibration Process")

print("Setting CONFIG mode")
set_mode(CONFIG_MODE)

print("loading calibration data...")
cal = load_calibration()
write_calibration(cal)

print("Entering NDOF_FMC_OFF mode to lock calibration...")
set_mode(NDOF_FMC_OFF_MODE)

print("Format - Euler angles - (Heading, Roll, Pitch)")
print("Format - Quaternion - (w, x, y, z)")