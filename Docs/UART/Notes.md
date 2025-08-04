# Project README: Raspberry Pi to Arduino UART Communication

This document serves as a log and technical guide for the UART communication project between a Raspberry Pi 4 and an Arduino Nano.

## 1. Project Setup and Configuration

### 1.1. Directory Structure
The primary project directory is located at `/home/pi/UART_protocol`.

### 1.2. Python Virtual Environment
To maintain dependency isolation, a Python virtual environment has been established within the project directory.

*   **Creation Command:**
    ```bash
    python3 -m venv uart_env
    ```
*   **Activation:** The environment must be activated for each new terminal session to ensure the correct Python interpreter and libraries are used.
    ```bash
    source uart_env/bin/activate
    ```

### 1.3. Raspberry Pi Hardware Configuration
By default, the Raspberry Pi's high-performance UART is assigned to the onboard Bluetooth module. To assign this UART to the GPIO pins for maximum performance and stability, the following line was added to `/boot/config.txt`:
dtoverlay=miniuart-bt

A system reboot is required for this change to take effect.

## 2. Raspberry Pi Communication Protocols

The following table summarizes the access methods for common communication protocols on the Raspberry Pi 4.

| Protocol | Enable in `raspi-config`? | Linux Device Path |
| :--- | :--- | :--- |
| **UART** | Yes (under Interface Options) | `/dev/serial0`, `/dev/ttyAMA0` |
| **I²C** | Yes | `/dev/i2c-1` |
| **SPI** | Yes | `/dev/spidev0.0`, `/dev/spidev0.1` |
| **USB** | No need (automatic) | `/dev/ttyUSBx`, `/dev/video0`, etc. |
| **CAN** | Requires overlay + transceiver | `can0` (via SocketCAN) |

## 3. Stage 1: Initial Communication Test

A basic "echo" test was performed to verify the physical UART connection and software setup. The communication link is confirmed to be working correctly.

### 3.1. Arduino Echo Sketch
This sketch listens for any incoming byte on the hardware serial port and immediately sends the exact same byte back.

```cpp
// ARDUINO
void setup() {
  // Initialize the serial port. The baud rate (9600) must match the Raspberry Pi script.
  Serial.begin(9600); 
  Serial.println("Arduino Echo is Ready."); // This message goes to your PC, not the Pi.
}

void loop() {
  // Check if there is any incoming data from the Raspberry Pi
  if (Serial.available() > 0) {
    // Read the incoming byte
    byte incomingByte = Serial.read();

    // Send the exact same byte back to the Raspberry Pi
    Serial.write(incomingByte);
  }
}
```
### 3.2. Raspberry Pi Ping Script
This script sends a test string to the Arduino and verifies if the received echo matches the original message.

```cpp
# RASPBERRY PI
import serial
import time

# --- Configuration ---
SERIAL_PORT = '/dev/serial0'
BAUD_RATE = 9600

# --- Main Program Logic ---
ser = None
try:
    print("Initializing serial connection...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
    time.sleep(2)

    message_to_send = b"Hello from your Pi!"
    print(f"Sending: '{message_to_send.decode()}'")
    ser.write(message_to_send)

    print("Waiting for echo...")
    echoed_message = ser.read(len(message_to_send))

    if echoed_message == message_to_send:
        print("\n--- SUCCESS ---")
        print(f"The Arduino correctly echoed back: '{echoed_message.decode()}'")
    elif echoed_message:
        print("\n--- FAILURE ---")
        print(f"Expected '{message_to_send.decode()}' but received '{echoed_message.decode()}' instead.")
    else:
        print("\n--- FAILURE ---")
        print("No response received from the Arduino.")

except serial.SerialException as e:
    print(f"\n[ERROR] Could not open or use the serial port: {e}")
finally:
    if ser and ser.is_open:
        ser.close()
        print("\nSerial port closed.")
```

### 4. Stage 2: Custom Quaternion Protocol
The next phase of the project is to implement a custom binary protocol for reliably transmitting quaternion data from the Raspberry Pi to the Arduino.
4.1. Arduino Quaternion Receiver Sketch
This sketch listens for a specific 18-byte packet, verifies its integrity with a checksum, and unpacks the payload into a quaternion data structure using a union for efficiency.

```cpp
// ARDUINO
/*
  Quaternion Binary Receiver
*/
typedef union {
  struct {
    float w, x, y, z;
  } q;
  byte bytes;
} QuaternionUnion;

QuaternionUnion receivedQuaternion;

void setup() {
  Serial.begin(9600); 
  Serial.println("Arduino Quaternion Receiver Ready.");
}

void loop() {
  // Check if a full packet is available to be read (start byte + 16 bytes payload + checksum byte)
  if (Serial.available() >= 18) {
    // 1. Look for the start byte to sync up.
    if (Serial.read() == 0xAA) { // Our chosen start byte
      
      // 2. Read the 16 bytes of the quaternion payload into our union.
      Serial.readBytes(receivedQuaternion.bytes, 16);

      // 3. Read the checksum byte sent by the Pi.
      byte receivedChecksum = Serial.read();

      // 4. Calculate our own checksum from the data we received.
      byte calculatedChecksum = 0;
      for (int i = 0; i < 16; i++) {
        calculatedChecksum += receivedQuaternion.bytes[i];
      }

      // 5. Verify if the checksums match.
      if (receivedChecksum == calculatedChecksum) {
        // Success! The data is valid. Print it.
        printQuaternion();
      } else {
        // Data was corrupted during transmission.
        Serial.println("Checksum mismatch! Packet dropped.");
      }
    }
  }
}

void printQuaternion() {
  Serial.print("Received OK: [w=");
  Serial.print(receivedQuaternion.q.w, 4);
  Serial.print(", x=");
  Serial.print(receivedQuaternion.q.x, 4);
  Serial.print(", y=");
  Serial.print(receivedQuaternion.q.y, 4);
  Serial.print(", z=");
  Serial.print(receivedQuaternion.q.z, 4);
  Serial.println("]");
}
```

### 4.2. Raspberry Pi Quaternion Sender Script
This script continuously generates sample quaternion data, packs it into the custom 18-byte packet format (Start Byte + 16-byte Payload + 1-byte Checksum), and transmits it to the Arduino.

```cpp
# RASPBERRY PI
import serial
import time
import struct

SERIAL_PORT = '/dev/serial0'
BAUD_RATE = 9600

ser = None
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE)
    print("Serial port opened. Waiting for Arduino to be ready...")
    time.sleep(2)
    print("Ready to send quaternion data.")

    angle = 0
    while True:
        w = 0.707
        x = float(angle)
        y = 0.5
        z = 0.0

        payload = struct.pack('<4f', w, x, y, z)
        checksum = sum(payload) & 0xFF
        packet = b'\xAA' + payload + struct.pack('<B', checksum)

        ser.write(packet)
        print(f"Sent: w={w:.2f}, x={x:.2f}, y={y:.2f}, z={z:.2f}")

        angle += 1
        if angle > 360:
            angle = 0

        time.sleep(0.5)

except serial.SerialException as e:
    print(f"ERROR: {e}")
except KeyboardInterrupt:
    print("\nProgram stopped by user.")
finally:
    if ser and ser.is_open:
        ser.close()
        print("Serial port closed.")
```