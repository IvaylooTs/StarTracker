from flask import Flask, Response
from threading import Event
from Flare import print_header
from flask_cors import CORS
from picamera2 import Picamera2
import cv2
import threading



app = Flask(__name__)
CORS(app)  
stop_event = Event()

print_header("Starting Picamera2")
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (1920, 1080)}))
picam2.start()

@app.route('/capture_photo')
def capture_photo():
    # Capture one frame from the camera
    frame = picam2.capture_array()
    # request = picam2.capture_request()
    # frame = request.make_array()
    # request.release()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Encode the frame to JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    # Return the image as a response with image/jpeg MIME type
    return Response(buffer.tobytes(), mimetype='image/jpeg')

def return_photo():
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # Encode the frame to JPEG
    _, buffer = cv2.imencode('.jpg', frame)
    # Return the image as a response with image/jpeg MIME type
    # return Response(buffer.tobytes(), mimetype='image/jpeg')
    return buffer.tobytes()

def gen_frames():
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # frame = cv2.flip(frame, -1)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def run_flask():
    app.run(host='192.168.55.160', port=5000, ssl_context=('cert.pem', 'key.pem'))
def signal_handler(sig, frame):
    print("\nStopping server and main loop...")
    stop_event.set()
    sys.exit(0)
