# webscoket.py
import asyncio
import websockets

from IMU import *
from SystemStats import *
from LostInSpace import lost_in_space
from FlaskServer import *
from Flare import *

def handleCommand(cmd):

    print_info(f"Received command: {cmd}")
    # jsonfy the command
    cmd = json.loads(cmd)
    print(cmd)
    # make multiple choises here based on the command
    # without using if statements
    match cmd.get("action"):
        case "reset":
            print_info("Resetting sensor...")
            # Reset logic here
        case "calibrate":
            print_info("Calibrating sensor...")
            StartCalibration()
            # Calibration logic here
        case "reboot":
            print_info("System reboot!")
            # Reboot logic here
        case "addOffset":
            print_info("Adding offset...")
            quaternion = cmd.get("data")
            print(quaternion)
            AddedQuaternionOffset = (quaternion["w"], quaternion["x"], quaternion["y"], quaternion["z"])
            AddOffsetToQuaternion(AddedQuaternionOffset)
            StartCalibration()
        case "lostInSpaceTest":
            print_info("Lost in space... with test images")
            quaternion = lost_in_space()   
            print_info(f"Lost in space quaternion: {quaternion}")   
        case "lostInSpaceCamera":
            print_info("Lost in space... with camera")
            file_location = save_photo_locally()
            try:
                quaternion = lost_in_space(file_location)   
                print_info(f"Lost in space quaternion: {quaternion}")   
            except Exception as e:
                print_error(f"lost in space failed {e}")
            finally:
                print_warning(f"Continue normal operation")
        case _:
            print_error(f"Unknown command: {cmd}")

async def handler(websocket):
    global cpu_temp,cpu_usage,ram_usage
    print_info("Client connected")
    try:
        # w, x, y, z = read_quaternion()
        # inverse = read_quaternion()
        # if (w,x,y,z) != (0,0,0,0):
        #     while (inverse == (0, 0, 0, 0)):
        #         w, x, y, z = read_quaternion()
        #         print(f"Initial Quaternion: w={w}, x={x}, y={y}, z={z}")
        #         await asyncio.sleep(0.1)
        #         if (w,x,y,z) != (0,0,0,0):
        #             inverse = inverse_quaternion(w, x, y, z)

        async def receive_commands():
            try:
                async for message in websocket:
                    print_info(f"Received from client: {message}")
                    handleCommand(message)  # Process the command
            except websockets.exceptions.ConnectionClosedError as e:
                print_error(f"Connection closed unexpectedly: {e}")
            except Exception as e:
                print_fatal_error(f"Other error: {e}")
          
        asyncio.create_task(receive_commands())
        # qw_inv, qx_inv, qy_inv, qz_inv = inverse 
        while True:

            w, x, y, z = Quaternion_After_Proceessing()

            new_w = w
            new_x = x
            new_y = y
            new_z = z
            heading, roll, pitch = (0,0,0)#read_euler()
          
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
            await asyncio.sleep(0.05)
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def StartWebSocket():
    print_info("Starting WebSocket server...")
    async with websockets.serve(handler, ip, 6789):
        print_success(f"WebSocket server started on ws://{ip}:6789")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(StartWebSocket())