from flask import Flask, render_template, Response, jsonify
from picamera2 import Picamera2
import cv2
import time
import board
import busio
import adafruit_bno055

# --- Flask App Setup ---
from imutils.video import VideoStream

app = Flask(__name__)

# --- Camera Setup ---
picam2 = Picamera2()
picam2.configure(picam2.create_video_configuration(main={"size": (640, 480)}))
picam2.set_controls({
    "FrameDurationLimits": (33333, 33333),  # 30 fps
    "ExposureTime": 10000,  # 10 ms
    "AnalogueGain": 1.0,
    "AwbEnable": False,
    "AeEnable": False,
    "AwbMode": 0,
})
picam2.start()
time.sleep(1)  # Let camera warm up

# --- BNO055 Sensor Setup ---
i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_bno055.BNO055_I2C(i2c)
def return_data():
    return sensor.euler

time.sleep(1)
sensor.mode = adafruit_bno055.CONFIG_MODE
time.sleep(0.5)
sensor.mode = adafruit_bno055.NDOF_MODE
time.sleep(0.5)
print("BNO055 Sensor Initialized.")


cal = sensor.calibration_status
sys, gyro, accel, mag = cal
print(f"Calibration Status - Sys: {sys}, Gyro: {gyro}, Accel: {accel}, Mag: {mag}")

# --- Rotation Data (Global Variable) ---
rotation_data = {"euler": None, "quaternion": None, "temperature": None}

# --- Frame Generator ---
def gen_frames():
    global rotation_data
    while True:
        frame = picam2.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.flip(frame, -1)

        # --- Read Sensor Data ---
        try:
            euler = sensor.euler
            quat = sensor.quaternion
            temp = sensor.temperature

            rotation_data = {
                "euler": euler,
                "quaternion": quat,
                "temperature": temp
            }

            # Optional: Draw Euler Angles on Frame
            if euler:
                text = f"Euler: {euler}"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        except Exception as e:
            print(f"Sensor error: {e}")

        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- Flask Routes ---
@app.route('/')
def index():
    return '''
    <html>
        <head><title>Raspberry Pi Camera</title></head>
        <body>
            <h1>Live Camera Stream</h1>
           <div style="
  display: flex;
  flex-wrap: wrap;">
            <img style="" src="/video_feed">
            <h2>Rotation Data</h2>
            <pre style="" id="rotation-data">Loading...</pre>
           </div>
             <button onclick="calibrateGyro()">Click Me</button>
             <p id="status-msg"></p>
            <script>
                setInterval(async () => {
                    const res = await fetch("/rotation");
                    const data = await res.json();
                    document.getElementById("rotation-data").textContent = JSON.stringify(data, null, 2);
                }, 300);
                async function calibrateGyro() {
                    const res = await fetch("/calibrate", {
                        method: "POST"
                    });
                    const data = await res.json();
                    document.getElementById("status-msg").textContent = data.message;
                }
            </script>
        </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/rotation')
def rotation():
    return jsonify(rotation_data)

@app.route('/calibrate', methods=['POST'])
def calibrate():
    print("Button was clicked!")
    return jsonify({"status": "success", "message": "Calibration is on {}!".format(return_data())})



# --- Main ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
