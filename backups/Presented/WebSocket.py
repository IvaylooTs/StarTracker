# webscoket.py
import asyncio
import websockets
from threading import Event

from IMU import *
from SystemStats import *
# from LostInSpace import lost_in_space
from LostInSpaceWrapper import get_current_lis_state
from FlaskServer import *
from Flare import *
from CommunicationInterface import *
from QuaternionPasser import *
from Math import *
from Mods import *

clients = set()  

def handleCommand(cmd):

    print_info(f"Received command: {cmd}")
    # jsonfy the command
    cmd = json.loads(cmd)
    print(cmd)
    # make multiple choises here based on the command
    # without using if statements
    match cmd.get("action"):
        case "rebootProgram":
            print_info("Rebooting program...")
            RebootProgram()
        case "rebootSystem":
            print_info("Rebooting program...")
            FullSystemRebootSequence()
        case "calibrate":
            print_info("Calibrating sensor...")
            ActiveCalibrationSequence()
        case "reboot":
            print_info("System reboot!")
        case "shutdown":
            FullSystemRebootSequence()
        case "lastCalibrationIMU":
            lastCalib = getLastCalibration()
            msg = f"Is in calibration range: {is_IMU_calibrated_recently()}, last calibration of IMU was {round(time.time() - lastCalib,3)} seconds ago"
            if(lastCalib == -1):
                msg = f"The IMU was never calibrated"
            print_info(msg)
            return ackSend(msg,cmd)
        case "addOffset":
            print_info("Adding offset...")
            quaternion = cmd.get("data")
            print(quaternion)
            AddedQuaternionOffset = (quaternion["w"], quaternion["x"], quaternion["y"], quaternion["z"])
            AddQuaternionAsOffsetSequence(AddedQuaternionOffset)
        case "lostInSpaceTest":
            print_info("Lost in space... with test images")
            if request_lost_in_space(None):
                print_info("Please wait for algorithm to finish")
            else:
                print_warning("Lost in space is running, please wait")
        case "lostInSpaceCamera":
            print_info("Lost in space... with camera")
            file_location = save_photo_locally()
            try:
                if request_lost_in_space(file_location):
                    print_info("Please wait for algorithm to finish")
                else:
                    print_warning("Lost in space is running, please wait")
            except Exception as e:
                print_error(f"lost in space failed {e}")
            finally:
                print_warning(f"Continue normal operation")
        case "getCalibrationQuaternions":
            return returnCalibrationInfo("",cmd)
        case "setReturnMode":
            mode = cmd.get("mode", "").strip().lower()
            if mode in ('auto', 'manual'):
                set_return_mode(mode)
            else:
                return errorSend(f"Invalid argument, {mode}",cmd)
        case "setQuaternionSource":
            source = cmd.get("source", "").strip().lower()
            if source in ("tracking", "imu", "lostinspace"):
                set_manual_source(source)
            else:
                return errorSend(f"Invalid argument, {source}", cmd)
            return ackSend(f"Mode changed to {source}", cmd)
        case "crazy":
            return errorSend("crazy? I was crazy once", cmd)
        case _:
            print_error(f"Unknown command: {cmd}")
            return errorSend("Unknown command",cmd)
    return ackSend("Accepted command!",cmd)

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

    angle = rotational_angle_between_quaternions(current_q, old_q)
    print_info(f"Angle between calibration is {angle}")

    data = {
            "calibrationInfo":{
                "angle": angle,
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

def get_quaternion_info():
    quat, trustFactor =passed_Quaternion()
    w, x, y, z = quat


    current_q, old_q = GetLastAddedOffset()
    c_w, c_x, c_y, c_z = current_q
    o_w, o_x, o_y, o_z = old_q

    angle = rotational_angle_between_quaternions(current_q, old_q)

    heading, roll, pitch = (0,0,0) #read_euler()
    
    temp =getTemp()
    cpu = getCpuUsage()
    ram = getRamUsage()

    # print_info(f"CPU_Temp: {temp} CPU_usage: {cpu} RAM_percent: {ram}")
    data = {
        "euler": {
            "heading": heading,
            "roll": roll,
            "pitch": pitch
        },
        "trust": trustFactor,
        "quaternion": {
            "w": w,
            "x": x,
            "y": y,
            "z": z
        },
        "stats":{

        "CPU_temp":  temp,
        "CPU_usage":  cpu,
        "RAM_percent": ram,
        },
        "currentMode": get_return_mode(),
        "lostInSpaceState": get_current_lis_state(),
        "angleDiff": angle
    }
    return data

async def handler(websocket):
    global cpu_temp,cpu_usage,ram_usage
    clients.add(websocket)

    client_ip = websocket.remote_address[0] if websocket.remote_address else "Unknown"
    client_port = websocket.remote_address[1] if websocket.remote_address else "Unknown"
    print_info(f"Clinet connected. IP - {client_ip}:{client_port}. Count of clients: {len(clients)}")
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
            data =get_quaternion_info()
            
            await websocket.send(json.dumps(data))
            await asyncio.sleep(0.040)
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    finally:
        clients.remove(websocket)
        print_info(f"Client left. Remaning clients: {len(clients)}")

async def StartWebSocket():
    print_info("Starting WebSocket server...")
    async with websockets.serve(handler, GetIP(), 6789):
        print_success(f"WebSocket server started on ws://{GetIP()}:6789")
        while not shutdown_is_set():
            await asyncio.sleep(0.1)
        print_warning("[INFO] Closing all client connections...")
        for ws in list(clients):
            await ws.close()
    print_success("Stopped websockets")


def StopWebSocket():
    # shutdown_event.set()
    shutdown_interupt()


if __name__ == "__main__":
    asyncio.run(StartWebSocket())
