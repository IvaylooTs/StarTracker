import shutil
import datetime
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# ASCII Art Header
text="""

 ▗▄▄▖▗▄▄▄▖▗▄▖ ▗▄▄▖     ▗▄▄▄▖▗▄▄▖  ▗▄▖  ▗▄▄▖▗▖ ▗▖▗▄▄▄▖▗▄▄▖      ▗▄▄▖ ▗▄▖ ▗▄▄▄▖▗▄▄▄▖▗▖ ▗▖ ▗▄▖ ▗▄▄▖ ▗▄▄▄▖     
▐▌     █ ▐▌ ▐▌▐▌ ▐▌      █  ▐▌ ▐▌▐▌ ▐▌▐▌   ▐▌▗▞▘▐▌   ▐▌ ▐▌    ▐▌   ▐▌ ▐▌▐▌     █  ▐▌ ▐▌▐▌ ▐▌▐▌ ▐▌▐▌        
 ▝▀▚▖  █ ▐▛▀▜▌▐▛▀▚▖      █  ▐▛▀▚▖▐▛▀▜▌▐▌   ▐▛▚▖ ▐▛▀▀▘▐▛▀▚▖     ▝▀▚▖▐▌ ▐▌▐▛▀▀▘  █  ▐▌ ▐▌▐▛▀▜▌▐▛▀▚▖▐▛▀▀▘     
▗▄▄▞▘  █ ▐▌ ▐▌▐▌ ▐▌      █  ▐▌ ▐▌▐▌ ▐▌▝▚▄▄▖▐▌ ▐▌▐▙▄▄▖▐▌ ▐▌    ▗▄▄▞▘▝▚▄▞▘▐▌     █  ▐▙█▟▌▐▌ ▐▌▐▌ ▐▌▐▙▄▄▖     

"""

# Config
display_time = True
use_colors = True

# ----------------------------
# Terminal Utilities
# ----------------------------

def colorize(text, color_code):
    return f"\033[{color_code}m{text}\033[0m" if use_colors else text

def get_terminal_width():
    return shutil.get_terminal_size((80, 20)).columns

def print_centered(text):
    terminal_width = get_terminal_width()
    for line in text.split('\n'):
        print(line.center(terminal_width))

def get_time_stamp():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] if display_time else ""

def print_header(text):
    width = get_terminal_width()
    top_border    = "┌" + "─" * (width - 2) + "┐"
    middle_line   = f"│{text.center(width - 2)}│"
    bottom_border = "└" + "─" * (width - 2) + "┘"

    print(colorize(top_border, "1;32"))
    print(colorize(middle_line, "1;32"))
    print(colorize(bottom_border, "1;32"))

# ----------------------------
# Logger Setup
# ----------------------------

def init_logger(log_dir="/home/pi/Week2/Comms/logs", log_level=logging.DEBUG):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = os.path.join(log_dir, f"log_{timestamp}.txt")

    logger = logging.getLogger("AppLogger")
    logger.setLevel(log_level)
    logger.handlers.clear()

    # File handler
    file_handler = RotatingFileHandler(log_file_path, maxBytes=5_000_000, backupCount=3)
    file_formatter = logging.Formatter('%(asctime)s  %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler
    # stream_handler = logging.StreamHandler(sys.stdout)
    # stream_handler.setFormatter(logging.Formatter('%(message)s'))
    # logger.addHandler(stream_handler)

    return logger

# Initialize logger
logger = init_logger()

# ----------------------------
# Logging with Colors
# ----------------------------

def log(label, text, color_code="1;37", level="INFO"):
    time_part = f"{get_time_stamp()} " if display_time else ""
    label_formatted = f"| {label:<7} |"

    colored_output = f"{colorize(time_part, '1;36')}{colorize(label_formatted, color_code)} {colorize(text, '1;37')}"
    plain_output   = f"{label_formatted} {text}"

    print(colored_output)

    if level == "INFO":
        logger.info(plain_output)
    elif level == "WARNING":
        logger.warning(plain_output)
    elif level == "ERROR":
        logger.error(plain_output)
    elif level == "FATAL":
        logger.critical(plain_output)
    else:
        logger.debug(plain_output)

# ----------------------------
# Log Type Wrappers
# ----------------------------

def print_info(text):        log("Info", text, "1;34", level="INFO")
def print_warning(text):     log("Warning", text, "1;33", level="WARNING")
def print_error(text):       log("Error", text, "1;31", level="ERROR")
def print_fatal_error(text): log("Fatal", text, "0;31", level="FATAL")
def print_success(text):     log("Success", text, "1;32", level="INFO")
# ----------------------------
# Print Banner
# ----------------------------

print_centered(text)
