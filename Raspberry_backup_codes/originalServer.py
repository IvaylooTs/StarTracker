

from flask import Flask, render_template, Response
from picamera2 import Picamera2
import cv2
import time

app = Flask(__name__)
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"format": 'XRGB8888',"size": (640, 480)}))
picam2.start()
time.sleep(1)  # Give time to warm up

def gen_frames():
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame = cv2.flip(frame, -1)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return '''
    <html>
        <head><title>Raspberry Pi Camera</title></head>
        <body>
            <h1>Live Camera Stream</h1>
            <img src="/video_feed">
        </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
