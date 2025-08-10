import serial
import time
import struct

from Flare import *
from IMU import Quaternion_After_Proceessing
from Math import round_quaternion
# --- Configuration ---
SERIAL_PORT = '/dev/serial0'
# Baud rate MUST match the Arduino sketch
BAUD_RATE = 9600

serial_object = None
def initSerial():
    global serial_object
    try:
        serial_object = serial.Serial(SERIAL_PORT, BAUD_RATE)
        return True
    except serial.SerialException as e:
        print_fatal_error(f"ERROR: {e}")
        serial_object = None
        return False
    except KeyboardInterrupt:
        print_error("\nProgram stopped by user.")
        serial_object = None
        return False

def startSerial():
    print_info("Starting serial communications...")
    if serial_object is None:
        print_error("No serial connection is initialized before calling startSerial().")
        return False
    while True:
        try:
            (w, x, y, z) = round_quaternion(Quaternion_After_Proceessing(), 3)

            payload = struct.pack('<4f', w, x, y, z)
            checksum = sum(payload) & 0xFF
            packet = b'\xAA' + payload + struct.pack('<B', checksum)

            time.sleep(0.1)
            serial_object.write(packet)

        except serial.SerialException as e:
            print_error(f"Serial device disconnected: {e}")
            break  # Exit loop if disconnected

        except Exception as e:
            print_error(f"Unexpected error: {e}")
