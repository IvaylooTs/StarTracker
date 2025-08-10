import psutil
import os
import time

ip = "192.168.55.160"

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp_str = f.read()
        return float(temp_str) / 1000.0
    except FileNotFoundError:
        return None

def get_cpu_usage():
    return psutil.cpu_percent(interval=1)

def get_ram_usage():
    mem = psutil.virtual_memory()
    return {
        "total": round(mem.total / (1024 ** 2), 2),  # in MB
        "used": round(mem.used / (1024 ** 2), 2),
        "percent": mem.percent
    }

cpu_temp = 0
cpu_usage = 1
ram_usage = 2

def getTemp():
    global cpu_temp
    return cpu_temp
def getCpuUsage():
    global cpu_usage
    return cpu_usage
def getRamUsage():
    global ram_usage
    return ram_usage

def PeriodicDataGrab():
    global cpu_temp, cpu_usage, ram_usage
    while True:
        cpu_temp = get_cpu_temp()
        cpu_usage = get_cpu_usage()
        ram_usage = get_ram_usage()
        time.sleep(1)  # Adjust the sleep time as needed    