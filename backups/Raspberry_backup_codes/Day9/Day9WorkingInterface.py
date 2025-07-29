import asyncio
import sys
import websockets
from scipy.spatial.transform import Rotation as R
import time
import json
from smbus2 import SMBus
import threading
from threading import Event 
import signal


import Math

from flask import Flask, Response
import cv2
from picamera2 import Picamera2


ip ="192.168.55.160"


# from scipy.spatial.transform import Rotation as R
# import numpy as np
import math


# ============================================================
#                      Slag
# ============================================================

text="""


 ▗▄▄▖▗▄▄▄▖▗▄▖ ▗▄▄▖     ▗▄▄▄▖▗▄▄▖  ▗▄▖  ▗▄▄▖▗▖ ▗▖▗▄▄▄▖▗▄▄▖      ▗▄▄▖ ▗▄▖ ▗▄▄▄▖▗▄▄▄▖▗▖ ▗▖ ▗▄▖ ▗▄▄▖ ▗▄▄▄▖     
▐▌     █ ▐▌ ▐▌▐▌ ▐▌      █  ▐▌ ▐▌▐▌ ▐▌▐▌   ▐▌▗▞▘▐▌   ▐▌ ▐▌    ▐▌   ▐▌ ▐▌▐▌     █  ▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌▐▌        
 ▝▀▚▖  █ ▐▛▀▜▌▐▛▀▚▖      █  ▐▛▀▚▖▐▛▀▜▌▐▌   ▐▛▚▖ ▐▛▀▀▘▐▛▀▚▖     ▝▀▚▖▐▌ ▐▌▐▛▀▀▘  █  ▐▌ ▐▌▐▛▀▜▌▐▛▀▚▖▐▛▀▀▘     
▗▄▄▞▘  █ ▐▌ ▐▌▐▌ ▐▌      █  ▐▌ ▐▌▐▌ ▐▌▝▚▄▄▖▐▌ ▐▌▐▙▄▄▖▐▌ ▐▌    ▗▄▄▞▘▝▚▄▞▘▐▌     █  ▐▙█▟▌▐▌ ▐▌▐▌ ▐▌▐▙▄▄▖     
                                                                                                        
"""

print(text)

def print_header(text):
    lines = text.split('\n')
    max_length = max(len(line) for line in lines)
    print("\033[1;32m" + "=" * (max_length + 4) + "\033[0m")
    for line in lines:
        print("\033[1;32m" + f"| {line:<{max_length}} |" + "\033[0m")
    print("\033[1;32m" + "=" * (max_length + 4) + "\033[0m")


# ============================================================
#                       BNO055 Sensor Configuration 
# ============================================================

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
#  ============================================================
#                       Read and process quaternion
# ============================================================




# print("--------------------------------------------------------------")
# print(f"Norm = {norm:.4f}")  # Should be ~1.0
# print(f"Inverse Quaternion: w={qw_inv:.4f}, x={qx_inv:.4f}, y={qy_inv:.4f}, z={qz_inv:.4f}")
# print("Stabilization to identity:", stabilization)
# print("software offset",q_adjusted.as_euler('xyz', degrees=True))
# print('...................................................................`.')
# print(f"Quaternion: w={w:.4f}, x={x:.4f}, y={y:.4f}, z={z:.4f}")
# print(f"Sensor Euler Angles -- Heading: {heading:.2f}°, Roll: {roll:.2f}°, Pitch: {pitch:.2f}°")
# print()

# ===========================================================
#                       WebSocket Server
# ===========================================================

def handleCommand(cmd):
    print("Received command:", cmd)
    # jsonfy the command
    cmd = json.loads(cmd)
    print(cmd)
    # make multiple choises here based on the command
    # without using if statements
    match cmd.get("action"):
        case "reset":
            print("Resetting sensor...")
            # Reset logic here
        case "calibrate":
            print("Calibrating sensor...")
            # Calibration logic here
        case "reboot":
            print("System reboot!")
            # Reboot logic here
        case _:
            print("Unknown command:", cmd)

async def handler(websocket):
    print("Client connected")
    try:
        w, x, y, z = read_quaternion()
        inverse = read_quaternion()
        while(inverse == (0, 0, 0, 0)):
            w, x, y, z = read_quaternion()
            print(f"Initial Quaternion: w={w}, x={x}, y={y}, z={z}")
            await asyncio.sleep(0.1)
            inverse = inverse_quaternion(w, x, y, z)

        async def receive_commands():
            async for message in websocket:
                print("Received from client:", message)
                handleCommand(message)  # Process the command
                # Here you can parse JSON commands and act on them

        asyncio.create_task(receive_commands())
        qw_inv, qx_inv, qy_inv, qz_inv = inverse 
        while True:

            w, x, y, z = read_quaternion()

            # Normalize quaternion and calculate inverse
            # norm = math.sqrt(w**2 + x**2 + y**2 + z**2)

            stabilization = round_quaternion(multiply_quaternions((w, x, y, z), (qw_inv, qx_inv, qy_inv, qz_inv)))


            new_w = w
            new_x = x
            new_y = y
            new_z = z
            # print("quaternion:", w,x,y,z)
            # print("inv quaternion:", qw_inv, qx_inv, qy_inv, qz_inv)
            # q_current = R.from_quat([x, y, z, w])  # Sensor reading
            # q_offset = q_current.inv()  
            # small_rotation = R.from_euler('y', 5, degrees=True)
            # q_adjusted = small_rotation * q_offset * q_current
            heading, roll, pitch = read_euler()
            # await websocket.send(f"'heading': {heading}, 'roll': {roll}, 'pitch': {pitch}")
            # await websocket.send(f"'quaternion': {{'w': {w}, 'x': {x}, 'y': {y}, 'z': {z}}}")

            # new_w = stabilization[0];
            # new_x = stabilization[1];
            # new_y = stabilization[2];
            # new_z = stabilization[3];

            data = {
                "euler": {
                    "heading": heading,
                    "roll": roll,
                    "pitch": pitch
                },
                "quaternion": {
                    "w": new_w,
                    "x": new_x,
                    "y": new_y,
                    "z": new_z
                }
            }

            await websocket.send(json.dumps(data))
            await asyncio.sleep(0.10)
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def main():
    async with websockets.serve(handler, ip, 6789):
        print(f"WebSocket server started on ws://{ip}:6789")
        await asyncio.Future()  # run forever

# Entry point



app = Flask(__name__)

stop_event = Event()

print_header("Starting Picamera2")
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.start()
time.sleep(1)  # Warm-up time


def gen_frames():
    while True:
        
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # frame = cv2.flip(frame, -1)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def run_flask():
    app.run(host='192.168.55.160', port=5000, ssl_context=('cert.pem', 'key.pem'))
    # app.run(host='192.168.55.160', port=5000)
# if __name__ == '__main__':

def start_servers():
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True  # Ensure thread exits when main program exits
    print_header("Starting Flask server")
    flask_thread.start()
    print_header("Starting WebSocket server")
    asyncio.run(main())

def signal_handler(sig, frame):
    print("\nStopping server and main loop...")
    stop_event.set()
    sys.exit(0)

    
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    try:
        start_servers()
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught in main")
    finally:
        print("Exiting program")
