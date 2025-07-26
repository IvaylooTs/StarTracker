>import time
from datetime import datetime
import json
from smbus2 import SMBus
import logging
import sys
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

Started_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

Log_filename = f"Log_{sys.argv[0]}_{Started_timestamp}.txt"

logging.basicConfig(
 filename=Log_filename,
 encoding='utf-8',
 level=logging.INFO,
 format='%(asctime)s - %(levelname)s - %(message)s')


console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

log = logging.getLogger(__name__)
log.addHandler(console_handler)

#logger.debug('This message should go to the log file')
#logger.info('So should this')
#logger.warning('And this, too')
#logger.error('And non-ASCII stuff, too, like Øresund and Malmö')

log.info(   " ----------------------- ")
log.info(   " Starting drift test.    ")
log.warning(" Don't move the sensor.  ")
log.info(   " ----------------------- ")
log.info(  f"  log file: {Log_filename}")

log.info(" >> Setting CONFIG mode to load calibration...")
set_mode(CONFIG_MODE)

cal = load_calibration()
write_calibration(cal)

log.info(" >> Entering NDOF_FMC_OFF mode to lock calibration...")
set_mode(NDOF_FMC_OFF_MODE)



log.info(" >> Reading Euler angles | format: (Heading, Roll, Pitch):")





try:
    while True:
        heading, roll, pitch = read_euler()
        w, x, y, z = read_quaternion()
        log.info(f"Quaternion: w={w:.4f}, x={x:.4f}, y={y:.4f}, z={z:.4f}")
        log.info(f"Heading: {heading:.2f}°, Roll: {roll:.2f}°, Pitch: {pitch:.2f}°")
        time.sleep(0.5)
except KeyboardInterrupt:
    log.warning("KeyboardInterrupt")
    log.info("\n End of program. Exiting...")
finally:
 Finish_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
 log.info( " ---------------------------- " )
 log.info(f" End time {Finish_time} " )
 log.info( " ---------------------------- " )
 log.info("Program ended")

