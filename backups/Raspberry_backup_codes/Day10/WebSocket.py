# webscoket.py
import asyncio
import websockets

from IMU import *
from SystemStats import *

ip = "192.168.55.160"

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
            StartCalibration()
            # Calibration logic here
        case "reboot":
            print("System reboot!")
            # Reboot logic here
        case _:
            print("Unknown command:", cmd)

async def handler(websocket):
    global cpu_temp,cpu_usage,ram_usage
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

            w, x, y, z = QuaternionAfterProceessing()

            # Normalize quaternion and calculate inverse
            # norm = math.sqrt(w**2 + x**2 + y**2 + z**2)

            # stabilization = round_quaternion(multiply_quaternions((w, x, y, z), (qw_inv, qx_inv, qy_inv, qz_inv)))


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
                },
                "CPU_temp":  getTemp(),
                "CPU_usage":  getCpuUsage(),
                "RAM_percent": getRamUsage()["percent"],
            }

            await websocket.send(json.dumps(data))
            await asyncio.sleep(0.025)
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def StartWebSocket():
    async with websockets.serve(handler, ip, 6789):
        print(f"WebSocket server started on ws://{ip}:6789")
        await asyncio.Future()  # run forever
