import time
import json
from smbus2 import SMBus

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

print("Setting CONFIG mode to load calibration...")
set_mode(CONFIG_MODE)

cal = load_calibration()
write_calibration(cal)

print("Entering NDOF_FMC_OFF mode to lock calibration...")
set_mode(NDOF_FMC_OFF_MODE)

print("Reading Euler angles (Heading, Roll, Pitch):")

try:
    while True:
        heading, roll, pitch = read_euler()
        print(f"Heading: {heading:.2f}°, Roll: {roll:.2f}°, Pitch: {pitch:.2f}°")
        time.sleep(0.5)
except KeyboardInterrupt:
    print("\nExiting...")
