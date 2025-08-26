import sys
import signal
import threading
import asyncio
import time

from Flare import print_header, print_warning
from WebSocket import StartWebSocket, StopWebSocket
from SystemStats import *
from IMU import InitIMU
from FlaskServer import run_flask, InitCamera
from UARTComms import *
from LostInSpaceWrapper import *

satellite_mode = False  # if true -> Desk mode

def start_flask_server():
    print_header("Starting Flask server")
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

def start_system_status_tracking():
    print_header("Starting System Stats Thread")
    sys_info_thread = threading.Thread(target=PeriodicDataGrab)
    sys_info_thread.daemon = True
    sys_info_thread.start()

def start_lost_in_space_thread():
    print_header("Starting lost in space thread")
    lis_thread = threading.Thread(target=
    lis_wrapper_auto_loop)
    lis_thread.daemon = True
    lis_thread.start()

def start_UART():
    if init_serial():
        # 2. If successful, create and start the listener thread.
        print_header("Starting UART communication listener")
        uart_thread = threading.Thread(target=listen_for_command)
        uart_thread.daemon = True
        uart_thread.start()
    else:
        # Use your custom print function for errors
        print_header("WARNING: UART communication failed to initialize and will be disabled.")

async def start_web_socket():
    print_header("Starting WebSocket server")
    await StartWebSocket()

async def start_servers():
    start_system_status_tracking()
    if not satellite_mode:
        start_flask_server()
        start_UART()
        asyncio.create_task(start_web_socket())  


def loop_shutdown():
    print_info("shuting down from singal loop asyncio")
    StopWebSocket()

def signal_handler(sig, frame):
    print("\nStopping server and main loop...")
    StopWebSocket()
    shutdown_interupt()
    sys.exit(get_exit_code())

async def main():
    try:
        await start_servers()
        start_lost_in_space_thread()
        InitCamera()
        InitIMU()
        while not shutdown_is_set():
            await asyncio.sleep(0.5)
    except KeyboardInterrupt:
        print_warning("KeyboardInterrupt caught in main")
    finally:
        print_warning("Exiting program")
        StopWebSocket()
        sys.exit(get_exit_code())  # Exit only after cleanup


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    asyncio.run(main())
