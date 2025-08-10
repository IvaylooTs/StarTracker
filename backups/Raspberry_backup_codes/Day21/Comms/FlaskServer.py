from flask import Flask, Response
from threading import Event
from Flare import *
from flask_cors import CORS
from picamera2 import Picamera2
import cv2
import threading
import time


high_res = (1920, 1080)
low_res = (640, 480)
res = high_res

app = Flask(__name__)
CORS(app)  
stop_event = Event()

picam2_object = None
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
            picam2_object.configure(picam2_object.create_video_configuration(main={"size": res}))
            picam2_object.start()
    except Exception as e:
        print_fatal_error(f"Failed to initialize Picamera2: {e}")
        picam2_object = None


@app.route('/capture_photo')
def capture_photo():
    global picam2_object
    # Capture one frame from the camera
    frame = picam2_object.capture_array()
    # reset colours to normal
    # ----------------------- this can be removed to save resources ---------------
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Encode the frame to JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    # Return the image as a response with image/jpeg MIME type
    return Response(buffer.tobytes(), mimetype='image/jpeg')

def return_photo():
    global picam2_object
    frame = picam2_object_object.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Encode the frame to JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    # Return the image as a response with image/jpeg MIME type
    # return Response(buffer.tobytes(), mimetype='image/jpeg')
    return buffer.tobytes()


def save_photo_locally(filename=None, format='jpg'):
    """
    Captures an image and saves it locally in JPG or PNG format.

    Parameters:
    - filename (str): Optional. If not provided, a timestamp-based name will be used.
    - format (str): 'jpg' or 'png'. Default is 'jpg'.

    Returns:
    - str: Absolute path of the saved image file.
    """
    global picam2_object
    frame = picam2_object.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Validate format
    format = format.lower()
    if format not in ['jpg', 'png']:
        raise ValueError("Format must be 'jpg' or 'png'")

    # Auto-generate filename if not provided
    if filename is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"photo_{timestamp}.{format}"
    else:
        # Ensure correct file extension
        base, _ = os.path.splitext(filename)
        filename = f"{base}.{format}"

    # Create full absolute path
    abs_path = os.path.abspath(filename)

    # Save the image
    success = cv2.imwrite(abs_path, frame)
    if not success:
        raise IOError("Failed to save the image.")

    print(f"Image saved as {abs_path}")
    return abs_path
    
def gen_frames():
    global picam2_object
    print_info("Starting video feed")

    # check for errors
    if picam2_object is None:
        print_error("Camera is not initialized.")
        return
    
    # start live feed
    while True:
        frame = picam2_object.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # frame = cv2.flip(frame, -1)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# server route for requesting a live feed
@app.route('/video_feed')
def video_feed():
    print_warning("Requested live video feed.")
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def run_flask():
    app.run(host='192.168.55.160', port=5000, ssl_context=('cert.pem', 'key.pem'))

# Handle closedown gracefully when kill is called (or ctrl+c)
def signal_handler(sig, frame):
    print_info("\nStopping server and main loop...")
    stop_event.set()
    sys.exit(0)
