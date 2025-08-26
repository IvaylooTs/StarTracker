from flask import Flask, Response
from threading import Event
from Flare import *
from flask_cors import CORS
from picamera2 import Picamera2
from SystemStats import *
import cv2
import threading
import time
import os
import sys
import signal
import numpy as np

# ---- Resolution ----
high_res = (1920, 1080)
low_res = (1366, 768)
res = high_res

# ---- Flask setup ----
app = Flask(__name__)
CORS(app)
stop_event = Event()

picam2_object = None

# ---- Camera Init ----
def InitCamera():
    global picam2_object
    print_header("Starting Picamera2")
    try:
        camera_list = Picamera2.global_camera_info()
        if not camera_list:
            print_error("No camera detected.")
            picam2_object = None
        else:
            picam2_object = Picamera2()
            picam2_object.configure(
                picam2_object.create_video_configuration(main={"size": res})
            )
            picam2_object.start()
    except Exception as e:
        print_fatal_error(f"Failed to initialize Picamera2: {e}")
        picam2_object = None


# ---- Helper: Draw Calibration Crosses ----
def draw_calibration_crosses(frame, color=(0, 255, 0), thickness=2, size=20):
    """
    Draws calibration crosses (center + edges) on the given frame.
    """
    h, w, _ = frame.shape
    points = [
        (w // 2, h // 2),  # center
        (w // 10, h // 10),  # top-left
        (w - w // 10, h // 10),  # top-right
        (w // 10, h - h // 10),  # bottom-left
        (w - w // 10, h - h // 10),  # bottom-right
    ]

    for (x, y) in points:
        # horizontal line
        cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
        # vertical line
        cv2.line(frame, (x, y - size), (x, y + size), color, thickness)

    return frame


# ---- Routes ----
@app.route("/capture_photo")
def capture_photo():
    global picam2_object
    frame = picam2_object.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", frame)
    return Response(buffer.tobytes(), mimetype="image/jpeg")


def return_photo():
    global picam2_object
    frame = picam2_object.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", frame)
    return buffer.tobytes()


def save_photo_locally(filename=None, format="jpg"):
    """
    Captures an image and saves it locally in JPG or PNG format.
    """
    global picam2_object
    frame = picam2_object.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    format = format.lower()
    if format not in ["jpg", "png"]:
        raise ValueError("Format must be 'jpg' or 'png'")

    if filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"photo_{timestamp}.{format}"
    else:
        base, _ = os.path.splitext(filename)
        filename = f"{base}.{format}"

    abs_path = os.path.abspath(filename)
    success = cv2.imwrite(abs_path, frame)
    if not success:
        raise IOError("Failed to save the image.")

    print(f"Image saved as {abs_path}")
    return abs_path


def gen_frames():
    global picam2_object
    print_info("Starting video feed")

    if picam2_object is None:
        print_error("Camera is not initialized.")
        return

    while True:
        frame = picam2_object.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # overlay calibration crosses
        frame = draw_calibration_crosses(frame)

        # _, buffer = cv2.imencode(".jpg", frame)
        _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 55])

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/video_feed")
def video_feed():
    print_warning("Requested live video feed.")
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/calibration_overlay")
def calibration_overlay():
    """
    Returns a calibration PNG with crosses for external display.
    """
    h, w = res[1], res[0]  # (height, width)
    frame = np.zeros((h, w, 3), dtype=np.uint8)  # black background
    frame = draw_calibration_crosses(frame, color=(0, 255, 0), thickness=2, size=30)

    _, buffer = cv2.imencode(".png", frame)
    return Response(buffer.tobytes(), mimetype="image/png")


# ---- Run Flask ----
def run_flask():
    ip = GetIP()
    print_info(f"Started flask server on ip: {ip}")
    app.run(host=ip, port=5000, ssl_context=("cert.pem", "key.pem"))


# ---- Signal Handler ----
def signal_handler(sig, frame):
    print_info("\nStopping server and main loop...")
    stop_event.set()
    sys.exit(0)


# ---- Main ----
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    InitCamera()
    run_flask()
