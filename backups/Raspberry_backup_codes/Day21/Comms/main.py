
import sys
from Flare import print_header 
from threading import Event
import signal
import threading
import asyncio

from WebSocket import StartWebSocket
from SystemStats import PeriodicDataGrab
from IMU import InitIMU

from UARTComms import startSerial, initSerial

# import FlaskServer
from FlaskServer import run_flask, InitCamera



stop_event = Event()


def start_servers():
    print_header("Starting System Stats Thread")
    sys_info_thread = threading.Thread(target=PeriodicDataGrab)
    sys_info_thread.daemon = True 
    sys_info_thread.start()

    print_header("Starting Flask server")
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True  # Ensure thread exits when main program exits
    flask_thread.start()
   

    # if(initSerial()):
    #     print_header("Starting UART communication")
    #     flask_thread = threading.Thread(target=startSerial)
    #     flask_thread.daemon = True  # Ensure thread exits when main program exits
    #     flask_thread.start()
   

    print_header("Starting WebSocket server")
    asyncio.run(StartWebSocket())


def signal_handler(sig, frame):
    print("\nStopping server and main loop...")
    stop_event.set()
    sys.exit(0)

    
if __name__ == "__main__":
    InitCamera()
    InitIMU()
    signal.signal(signal.SIGINT, signal_handler)
    try:
        start_servers()
    except KeyboardInterrupt:
        print("KeyboardInterrupt caught in main")
    finally:
        print("Exiting program")
