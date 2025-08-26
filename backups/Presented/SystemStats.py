import psutil
import os
import time
import threading
import subprocess
from Flare import *

DEFAULT_IP = "192.168.55.173"

def ip_for(interface, default=DEFAULT_IP):
    try:
        out = subprocess.check_output(
            ["ip", "-4", "addr", "show", interface], text=True
        )
        for line in out.splitlines():
            line = line.strip()
            if line.startswith("inet "):
                return line.split()[1].split("/")[0]
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to get IP for {interface}: {e}")
    except FileNotFoundError:
        print_error("The 'ip' command is not available on this system.")
    except Exception as e:
        print_error(f"Unexpected error while getting IP: {e}")
    return default

local_ip = ip_for("wlan0")

def GetIP():
    global local_ip
    return local_ip

shutdown_event = threading.Event()
exit_code_from_interupt = 0

def get_exit_code():
    global exit_code_from_interupt
    print_info(f"system info {exit_code_from_interupt}")
    return exit_code_from_interupt

def shutdown_interupt(exit_code=-1):
    global shutdown_event, exit_code_from_interupt
    if exit_code != -1:
        exit_code_from_interupt = exit_code
    print_info("ACTIVATING SHUTDOWN INTERUPT")
    shutdown_event.set()

def shutdown_is_set():
    global shutdown_event
    return shutdown_event.is_set()

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp_str = f.read()
        return float(temp_str) / 1000.0
    except FileNotFoundError:
        print_warn("CPU temperature file not found.")
        return None
    except PermissionError:
        print_error("Permission denied while reading CPU temperature.")
        return None
    except Exception as e:
        print_error(f"Unexpected error reading CPU temp: {e}")
        return None

def get_cpu_usage():
    try:
        return psutil.cpu_percent(interval=1)
    except Exception as e:
        print_error(f"Error fetching CPU usage: {e}")
        return None

def get_ram_usage():
    try:
        mem = psutil.virtual_memory()
        return {
            "total": round(mem.total / (1024 ** 2), 2),  # in MB
            "used": round(mem.used / (1024 ** 2), 2),
            "percent": mem.percent
        }
    except Exception as e:
        print_error(f"Error fetching RAM usage: {e}")
        return {"total": 0, "used": 0, "percent": 0}

cpu_temp = 0
cpu_usage = 1
ram_usage = {"percent": 0}

def getTemp():
    global cpu_temp
    return cpu_temp if cpu_temp is not None else 0

def getCpuUsage():
    global cpu_usage
    return cpu_usage if cpu_usage is not None else 0

def getRamUsage():
    global ram_usage
    return ram_usage.get("percent", 0)

def PeriodicDataGrab():
    global cpu_temp, cpu_usage, ram_usage
    while not shutdown_is_set():
        try:
            cpu_temp = get_cpu_temp()
            cpu_usage = get_cpu_usage()
            ram_usage = get_ram_usage()
        except Exception as e:
            print_error(f"Error in PeriodicDataGrab loop: {e}")
        time.sleep(1)  # Adjust the sleep time as needed
