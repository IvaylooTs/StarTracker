
import sys
from Flare import print_header 
from threading import Event
import signal
import threading
import asyncio

from WebSocket import StartWebSocket
from SystemStats import PeriodicDataGrab


# import FlaskServer
from FlaskServer import run_flask

ip ="192.168.55.160"


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
    print_header("Starting WebSocket server")
    asyncio.run(StartWebSocket())

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
