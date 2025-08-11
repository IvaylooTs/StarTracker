import sys
from Flare import print_header
from threading import Event
import signal
import threading
import asyncio
import time # Added for the UART loop sleep

# --- Your Existing Imports ---
from WebSocket import StartWebSocket
from SystemStats import PeriodicDataGrab
from IMU import InitIMU
from FlaskServer import run_flask, InitCamera

# --- Import the new, modular UARTComms ---
import UARTComms2

# --- Global Stop Event ---
stop_event = Event()

# --- New UART Thread Function ---
def uart_listener_thread():
    """
    This function is the target for our UART thread. It runs a continuous loop,
    polling for commands from the Arduino.
    """
    print_header("UART Listener Thread Started")
    while not stop_event.is_set():
        UARTComms2.listen_for_command()
        time.sleep(0.01)
    print("UART Listener Thread stopped.")


def start_servers():
    print_header("Starting System Stats Thread")
    sys_info_thread = threading.Thread(target=PeriodicDataGrab)
    sys_info_thread.daemon = True
    sys_info_thread.start()

    print_header("Starting Flask server")
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()
    # 1. Initialize the serial port first.
    if UARTComms2.init_serial():
        # 2. If successful, create and start the listener thread.
        print_header("Starting UART communication listener")
        uart_thread = threading.Thread(target=uart_listener_thread)
        uart_thread.daemon = True
        uart_thread.start()
    else:
        # Use your custom print function for errors
        print_header("WARNING: UART communication failed to initialize and will be disabled.")
    print_header("Starting WebSocket server (this will block)...")
    asyncio.run(StartWebSocket())


def signal_handler(sig, frame):
    print("\n[MAIN] Shutdown signal received. Stopping all services...")
    # Set the event to signal all threads to stop their loops
    stop_event.set()
    UARTComms2.close_serial() 
    
    time.sleep(1)
    
    sys.exit(0)


if __name__ == "__main__":
    InitCamera()
    InitIMU()

    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        start_servers()
    except KeyboardInterrupt:
        print("[MAIN] KeyboardInterrupt caught.")
    finally:
        # This block will run after start_servers() completes or after the signal_handler exits.
        print("[MAIN] Exiting program.")
