## Sensor Fusion Algorithm Design (e.g., Kalman Filter)

The main sensor fusion functionality is the combination of the Lost-in-space algorithm and IMU (inertial measurement unit) sensor data to calibrate and get accurate attitude data. There is another layer that can be explored to add redundancy in the system in case of hardware failure in specific sensor of the IMU.

for more information of the IMU check **IMU sensor - BNO055**
##### The problem with this external chip
Bosch kept the algorithms and firmware closed source. We cannot directly know what the chips is doing or change the code installed on it. 
#### Other possible solutions
We can still access the data from the sensors from their respective registers and shut off the computation logic. This can be useful in case of failure of the chip, algorithms or specific sensors. We can extract the information and do the computation on the main system and continue with the operation of the module with few limitations.
Right now this is in *research & development* state.

This is imagined to boot and check all needed registers for errors and reboot the sensor one time and then if this still fails, enter "sensor failure" mode, based on what stopped working. If everything is fine, we enter standard mode (without auto magnetic recalibration)
# Firmware 
Firmware is the main code that runs on the hardware and allows the integration of the software. Our system runs on Debian Linux 32-bit in headless mode.

The main program will boot and load needed libraries and subprograms for camera and IMU configuration, sensor fusion, data handling, communication, lost-in-space algorithms and database tables.

This operating system will run with limited system functionality - bluetooth, wifi, ethernet, most drivers and other unnecessary systems will be deactivated or deleted to save on space and efficiency.

It is important to note that Wifi is currently the only way to access the UI interafece and get information on the system. 
The {protocol} is used to allow the main computer to communicate with other subsystems and return quaternion information and receive commands for operation.

The protocols used for information transfers and direct controls are:
- SSH
- Websockets (2-way)
- Flask web server
### IMU sensor - BNO055
The current implementation uses the **bno055** for an IMU with integrated sensor fusion chip and proprietary Bosch algorithms for sensing and error correction in drift and other systems. It combines data from gyroscope, accelerometer and magnetometer, depending on the currently activated mode in the specific hardware register. Thanks to this chip we export the IMU data calculation on another system, so that the main computer (the Raspberry PI) has more free resources for other computation. 

TODO (for Tsetso): add statistics...
### Camera - Raspberry PI AI camera
This is one of the official Raspberry PI camera with 4056 x 3040 pixels (12.3 megapixels) resolution, based on the Sony IMX500. Even though this camera is designed for AI uses, It provides good photo quality and allows for faster and easier implementations. The AI chips allows for future developments and non-deterministic algorithms to be implemented.

TODO (for Tsetso): add statistics...
# Controls
To easily configure and work with the module, we provided a UI interface for control of the system and analysis. It gives easier way to visualize the information, see camera output and test specific functionalities (force calibration, change modes, etc..).
On the current version you can see 6 panels:
- 3D visualization of the quaternion data (using Three.js)
- Graph visualization of the quaternion data (using Graph.js)
- Received information in terminal
- Camera live* view (could be delays in slow computers or busy networks)
- General quaternion and system data
- Control buttons

The video feed is transferred via Flask web server hosted on the module and the data is received via websockets