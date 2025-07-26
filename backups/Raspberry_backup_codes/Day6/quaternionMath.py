import time
import json
import math
from smbus2 import SMBus
from scipy.spatial.transform import Rotation as R
import numpy as np

ADDR = 0x28
BUS = 1

OPR_MODE_ADDR = 0x3D
CALIBRATION_REG_START = 0x55
CALIBRATION_REG_LEN = 22

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

def read_euler():
    data = bus.read_i2c_block_data(ADDR, EULER_H_LSB, 6)  # 2 bytes each for heading, roll, pitch
    # convert from little endian signed 16-bit
    heading = int.from_bytes(data[0:2], byteorder='little', signed=True) / 16.0
    roll = int.from_bytes(data[2:4], byteorder='little', signed=True) / 16.0
    pitch = int.from_bytes(data[4:6], byteorder='little', signed=True) / 16.0
    return heading, roll, pitch

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

def inverse_quaternion(w, x, y, z):
    norm_sq = w**2 + x**2 + y**2 + z**2
    if norm_sq == 0:
        return 1, 0, 0, 0  # Avoid division by zero
    return (
        w / norm_sq,
        -x / norm_sq,
        -y / norm_sq,
        -z / norm_sq
    )
def multiply_quaternions(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return (w, x, y, z)


def round_quaternion(q, decimals=6):
    return tuple(round(c, decimals) for c in q)
# Convert quaternion to Euler angles


print("Setting CONFIG mode to load calibration...")
set_mode(CONFIG_MODE)

cal = load_calibration()
write_calibration(cal)

print("Entering NDOF_FMC_OFF mode to lock calibration...")
set_mode(NDOF_FMC_OFF_MODE)

print("Reading Euler angles (Heading, Roll, Pitch):")

offsetQuaternion = (1, 0, 0, 0) 

try:
    while True:

        # original readings
        heading, roll, pitch = read_euler()
        w, x, y, z = read_quaternion()

        # Normalize quaternion and calculate inverse
        norm = math.sqrt(w**2 + x**2 + y**2 + z**2)
        inverse = inverse_quaternion(w, x, y, z)
        qw_inv, qx_inv, qy_inv, qz_inv = inverse 

        stabilization = round_quaternion(multiply_quaternions((w, x, y, z), (qw_inv, qx_inv, qy_inv, qz_inv)))


        print("--------------------------------------------------------------")
        print(f"Norm = {norm:.4f}")  # Should be ~1.0
        print(f"Inverse Quaternion: w={qw_inv:.4f}, x={qx_inv:.4f}, y={qy_inv:.4f}, z={qz_inv:.4f}")
        print("Stabilization to identity:", stabilization)

        print('...................................................................`.')
        print(f"Quaternion: w={w:.4f}, x={x:.4f}, y={y:.4f}, z={z:.4f}")
        print(f"Sensor Euler Angles -- Heading: {heading:.2f}°, Roll: {roll:.2f}°, Pitch: {pitch:.2f}°")
        print()



        time.sleep(0.5)
except KeyboardInterrupt:
    print("\nExiting...")
