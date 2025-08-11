# UARTComms.py - Refactored as a Reusable Module

import serial
import time
import struct

# --- Your Custom Imports ---
# Assuming these functions exist and work as before
from IMU import Quaternion_After_Proceessing
from Math import round_quaternion

# --- Configuration ---
SERIAL_PORT = '/dev/serial0'
BAUD_RATE = 9600 # Make sure this matches the Arduino

# --- Module-level Global Variable ---
# This will hold our serial object for the module
serial_object = None

# --- Placeholder Print Functions ---
def print_info(msg):
    print(f"[INFO] {msg}")

def print_error(msg):
    print(f"[ERROR] {msg}")

# --- Public Functions (To be called from main.py) ---

def init_serial():
    """
    Initializes and opens the serial port. This should be called once at the
    start of your main program.
    Returns True on success, False on failure.
    """
    global serial_object
    if serial_object and serial_object.is_open:
        print_info("Serial port is already open.")
        return True
    try:
        serial_object = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1) # Non-blocking timeout
        print_info(f"Serial port {SERIAL_PORT} opened successfully at {BAUD_RATE} baud.")
        return True
    except serial.SerialException as e:
        print_error(f"Could not open serial port: {e}")
        serial_object = None
        return False

def listen_for_command():
    """
    Performs a single, non-blocking check for a command from the Arduino.
    This function should be called repeatedly in your main program's loop.
    It will handle the command if one is received.
    """
    if not (serial_object and serial_object.is_open):
        return # Do nothing if the port isn't open

    try:
        if serial_object.in_waiting > 0:
            # Read a line, decode, and strip whitespace
            incoming_command = serial_object.readline().decode('utf-8').strip()
            if incoming_command: # Make sure it's not an empty line
                print_info(f"Received command: {incoming_command}")
                _handle_command(incoming_command) # Call the private handler
    except serial.SerialException as e:
        print_error(f"Serial communication error: {e}")
        # You might want to add logic here to try and reconnect
        close_serial()


def close_serial():
    """
    Closes the serial port if it is open. Call this at the end of your main program.
    """
    global serial_object
    if serial_object and serial_object.is_open:
        serial_object.close()
        serial_object = None
        print_info("Serial port closed.")

# --- Private Helper Functions (Not intended to be called directly from main.py) ---

def _send_quaternion_response():
    """
    Private function to get, pack, and send the quaternion data.
    """
    try:
        (w, x, y, z) = round_quaternion(Quaternion_After_Proceessing(), 3)
        payload = struct.pack('<4f', w, x, y, z)
        checksum = sum(payload) & 0xFF
        packet = b'\xAA' + payload + struct.pack('<B', checksum)
        serial_object.write(packet)
        print_info(f"-> Responded with quaternion packet.")
    except Exception as e:
        print_error(f"Failed to send quaternion response: {e}")

def _handle_command(command):
    """
    Private function to parse the command and trigger the correct action.
    """
    if command == "<PING>":
        serial_object.write(b"[PONG]\n")
        print_info("-> Responded with PONG.")
    elif command == "<GET_QUAT>":
        _send_quaternion_response()
    else:
        # Silently ignore unknown commands
        pass
