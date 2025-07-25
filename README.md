# Repo: Star Tracker

**Unofficial Team Name:** 404: Star Not Found
**Unofficial Project Name:** CASSI (Celestial Alignment System for Satellite Inertial-control)

**Official category:** Star tracker 1

**Team Members:**  
Michael Yan, Ivaylo Tsekov, Alexandra Nedela, Dilyana Vasileva, Tsvetomir Staykov

---

## Warning!
This document is subject to change. It is suppost to show the general idea of the project. Ideas, concepts, implemnetation and code may change due to improvements, errors or obsolescence.

**Current status:** 
- check for errors in writing
- validate
- check for missing subjects
---

## Problem

When you are in space, you need attitude information. This is given with IMU, but problem comes when you try to calibrate the sensor. Due to accumulation of errors during computation and enviromental variables, effect called "drift" can be found.

## Project Overview 
This project implements a star tracking system using a **Raspberry Pi** with **(AI) camera** to determine the orientation of the system based on captured star images. Even thought the camera has AI accelerator capabilities, the system uses deterministic algorithm. Once calibrated, the system works in tandem with an IMU (Inertial Measurement Unit) to track orientation without continuous visual input. When drifting accures, due to enviromental variables, power loss or something else, the system automatically recalibrates. 

---

## Current available demo:

Go to folder `digital_twin`. It contains
- `main.py` - main files with all the systems
- `star_distances_sorted.db` - database with calculated angular distances and sorted in ascending order
- `star_identified.png` - sample image from previous tests
- `test_images` - good images from *Stellarium* for testing purposed. Uses specific resolution and FOV

To run the script you will need `sqlite3` and `cv2` libraries installed on you system and *python 3*

To run the script just run:
```bash
python3 main.py 
```
The results should be:
- valid stars that math the Angular distance.
- cordinates of stars found on the picture
- all angular distances that are close to the angular distances of the found stars
- candidates for stars in the python code in format `{star-number} -> [{star 1 hip}, {star 2 hip}, ... {star n hip}]`


---

## Current progress

#### Software
 - [x] Define project requirements and needed 
 - [x] Find Algorithm for identification of stars
 - [x] OpenCV star identification on image
  - [x] Test model with maths on hand
 - [x] Get angular distances between stars
 - [x] Make database of parameters for comparisons (star 1 HIP, star 2 HIP, angular distance)
 - [x] Select candidats for possible stars in specific location
 - [ ] Identify the correct stars
 - [ ] Find rotation matrix
 - [ ] Apply rotation vector to IMU measurements
 - [ ] Test for correctness  

#### Hardware
 - [X] Configure Raspberry pi
 - [X] Connect required modules
  - [X] Camera 
  - [X] IMU (Gyroscope, Accelerometer, Magnetometer)
 - [x] Calibrate on Hardware level IMU
  - [ ] Measure drift
  - [ ] Research drift correction 
 - [ ] Calibrate on Software level IMU
 - [ ] Add interface for communication with Arduino ( I2C is good candidate )
 - [ ] Develop firmware
  - [ ] auto boot configuration

#### CAD
 - [X] Construct simple box for fit testing and simple implementation
 - [X] Make box for better testing and protection
 - [ ] Satellite inclosure with removable modules
  - [ ] Down panel
  - [ ] sliders
  - [ ] test sliders
  - [ ] side protection panels
  - [ ] CASSI module 

#### Additional
 - [ ] Testing | Make black box for testing based on images on monitor
 - [ ] Visualization | Make control panel
  - [ ] 3d visualization
  - [ ] live video 
  - [ ] OpenCV calculations
  - [ ] Control buttons
 - [ ] Check if static ip is needed (just in case)

## System Architecture

### Catalogue Processing

- Filter stars based on:
  - Brightness
  - Size
  - Eccentricity
  - Angular Distance
- Select the optimal set of stars for pattern recognition
- Convert selected stars to Cartesian coordinates (stored in **Database Table 1**)
- Generate all possible star pairs and compute angles between them (stored in **Database Table 2**)

### Image Processing

- Filter stars in the image based on:
  - Brightness
  - Size
  - Eccentricity
- Detect approximately 3–20 stars from the image ( current tested number, possible improvements)
- Convert pixel coordinates to camera coordinates, then to Cartesian unit vectors
- Calculate angles between all star pairs ~~(stored in **Table 3**)~~

### Attitude Determination

- Match angles ~~from **Table 3**~~ with catalogue data in **Table 2** to identify observed stars
- Determine yaw, pitch, roll based on the known catalogue orientation (aligned with the celestial equator)
- Build transformation matrices via vector space basis change* (terminology needs validation) via:
  - Image star vectors (camera frame)
  - Catalogue star vectors (inertial frame)
<!-- - Compute the rotational transformation matrix to determine roll -->
- Convert the final orientation to *quaternions* (euler angles cause gimbal lock and unexpected value jumps)

---

## Hardware Integration

- The initial orientation determined via camera is stored as a reference
- The IMU is zeroed and calibrated at this orientation
- Subsequent orientation tracking is performed using only the IMU
- System will be designed to work without camera for some time (Still need camera for recalibration due to drift) 

**Final Orientation = IMU Output + Initial Camera-Based Offset**

---

## Interfacing
 - current iteration uses Wifi connection to host server to show data in json format and live server with camera and data overview
 - implementation of hardware protocol is used for communication between subsystems inside satellite (Arduino)

---

## Mechanical Design

- Modular, layered physical design:
  - Raspberry Pi
  - IMU module (Adafruit 9-DOF Absolute Orientation IMU Fusion Breakout with Bosch BNO055 Sensor)
  - Camera mounted perpendicular to the Raspberry Pi board (uses Sony IMX500 imaging sensor)
- 3D CAD models and assembly diagrams are available in the `DOCS/` directory

---

## Demonstration Instructions

1. Set up a dark environment using panels or an enclosure.
2. Point the device at a star image displayed on a screen with a known orientation.
3. Verify that the computed orientation matches the actual orientation.
4. Move the device and observe that orientation tracking continues using the IMU, even without camera input.

---
