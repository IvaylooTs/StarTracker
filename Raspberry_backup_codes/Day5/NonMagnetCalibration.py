import time
import json
from smbus2 import SMBus

ADDR = 0x28  # Or 0x29 depending on BNO055 ADR pin
BUS = 1

OPR_MODE_ADDR = 0x3D
CALIB_STAT = 0x35
CALIBRATION_REG_START = 0x55
CALIBRATION_REG_LEN = 22

CONFIG_MODE = 0x00
NDOF_FMC_OFF_MODE = 0x0C

bus = SMBus(BUS)

def set_mode(mode):
    bus.write_byte_data(ADDR, OPR_MODE_ADDR, CONFIG_MODE)
    time.sleep(0.05)
    bus.write_byte_data(ADDR, OPR_MODE_ADDR, mode)
    time.sleep(0.05)

def is_fully_calibrated():
    calib = bus.read_byte_data(ADDR, CALIB_STAT)
    sys = (calib >> 6) & 0x03
    gyro = (calib >> 4) & 0x03
    accel = (calib >> 2) & 0x03
    mag = calib & 0x03
    print(f"Calibration - Sys:{sys} Gyro:{gyro} Accel:{accel} Mag:{mag}")
    return sys == 3 and gyro == 3 and accel == 3 and mag == 3

def read_calibration():
    return bus.read_i2c_block_data(ADDR, CALIBRATION_REG_START, CALIBRATION_REG_LEN)

def save_calibration(data, filename="bno055_cal.json"):
    with open(filename, "w") as f:
        json.dump(data, f)
    print(f"Calibration saved to {filename}")

# Start
print("Setting mode to NDOF_FMC_OFF...")
set_mode(NDOF_FMC_OFF_MODE)

print("Move sensor to fully calibrate (figure-8, rotate)...")
while not is_fully_calibrated():
    time.sleep(1)

print("Fully calibrated!")
cal_data = read_calibration()
save_calibration(cal_data)
