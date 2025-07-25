import time
import board
import busio
import adafruit_bno055

i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_bno055.BNO055_I2C(i2c)

# Give sensor time to initialize
time.sleep(1)

# Optionally reset mode
sensor.mode = adafruit_bno055.CONFIG_MODE
time.sleep(0.5)
sensor.mode = adafruit_bno055.NDOF_MODE
time.sleep(0.5)

print("BNO055 Sensor Initialization Complete\n")

while True:
    try:
        euler = sensor.euler
        quat = sensor.quaternion
        temp = sensor.temperature

        print(f"Euler: {euler}")
        print(f"Quaternion: {quat}")
        print(f"Temperature: {temp} °C")
        print("--------------------")
        time.sleep(0.15)
    except Exception as e:
        print(f"Error reading sensor: {e}")
        time.sleep(2)
