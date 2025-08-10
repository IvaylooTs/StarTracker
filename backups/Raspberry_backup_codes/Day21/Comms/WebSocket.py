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
                x,y,z,w = quaternion
                AddOffsetToQuaternion((w,x,y,z))
                print_info(f"Lost in space quaternion: {w,x,y,z}")   
            except Exception as e:
                print_error(f"lost in space failed {e}")
            finally:
                print_warning(f"Continue normal operation")
        case "getCalibrationQuaternions":
            return returnCalibrationInfo("",cmd)
        case _:
            print_error(f"Unknown command: {cmd}")
            return errorSend("Unknown command",cmd)
    return ackSend("Good",cmd)

def ackSend(msg,cmd):
    data = {
            "ack":{
                "message":f"{msg}",
                "cmd": f"{cmd}"
            }
        }
    return data
def returnCalibrationInfo(msg,cmd):
    print_info("Requested calibration information.")
    current_q, old_q = GetLastAddedOffset()
    print_info(f"current: {current_q}")
    print_info(f"old: {old_q}")
    c_w, c_x, c_y, c_z = current_q
    o_w, o_x, o_y, o_z = old_q
    data = {
            "calibrationInfo":{
                "current": {
                    "w": f"{c_w}",
                    "x": f"{c_x}",
                    "y": f"{c_y}",
                    "z": f"{c_z}",
                },
                "old": {
                    "w": f"{o_w}",
                    "x": f"{o_x}",
                    "y": f"{o_y}",
                    "z": f"{o_z}",
                }
            }
        }
    return data
def errorSend(msg,cmd):
    data = {
            "error":{
                "message":f"{msg}",
                "cmd": f"{cmd}"
            }
        }
    return data
async def handler(websocket):
    global cpu_temp,cpu_usage,ram_usage
    print_info("Client connected")
    try:
        async def receive_commands():
            try:
                async for message in websocket:
                    print_info(f"Received from client: {message}")
                    response = handleCommand(message)  # Process the command
                    await websocket.send(json.dumps(response))
                    await asyncio.sleep(0.040)

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
                "stats":{

                "CPU_temp":  getTemp(),
                "CPU_usage":  getCpuUsage(),
                "RAM_percent": getRamUsage()["percent"],
                }
            }
            
            await websocket.send(json.dumps(data))
            await asyncio.sleep(0.040)
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

async def StartWebSocket():
    print_info("Starting WebSocket server...")
    async with websockets.serve(handler, ip, 6789):
        print_success(f"WebSocket server started on ws://{ip}:6789")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    asyncio.run(StartWebSocket())
