# Repo: Star Tracker

**Team Name:** 404: Star Not Found

**Unofficial Project Name:** CASSI (Celestial Alignment System for Satellite Inertial-control)

**Official category:** Star tracker 1

**Team Members:**  
Michael Yan, Ivaylo Tsekov, Alexandra Nedela, Dilyana Vasileva, Tsvetomir Staykov

---

## Warning!
This document is subject to change. It is suppost to show the general idea of the project. Ideas, concepts, implemnetation and code may change due to improvements, errors or obsolescence.

**Current status:** 
- check for errors in writing
- need validate
- may be missing subjects
---

## Problem

When you are in space, you need attitude information. This is given with IMU, the but problem comes when you try to calibrate the sensor. Due to accumulation of errors during computation and enviromental variables, effect called "drift" can be found. So how would we determine our correct attitude?

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
The results in your Terminal should be:
- pixel cordinates of stars found on the picture
- the angular distance of each pair (close to the real ones - needs to be optimized)
- candidates from the database for each star in our image 
- the image we're testing on should be displayed with the N brightest stars circled
## System Architecture

![Cool Diagram here](https://github.com/IvaylooTs/StarTracker/tree/main/Docs/Diagrams/AlgorithmDiagram.png)

### Software

#### Database structure
 We have two tables in our database. **Table 1 - "Stars"** contains the hip of each star according to the hipparcos catalog, the magnitude of each star and x, y, z coordinates of a unit vector coresponding to their RA and DEC coordinates in the equatorial coordinate system.
 **Table 2 - "AngularDistances"** contains the calculated angular distance between each pair of stars in Table 1 sorted from lowes to highest value of angular distance.
 The stars are filtered based on:
  - Brightness
  - Size
  - Eccentricity

#### Lost-in-Space algorithm
 We're implementing a Lost-in-Space approach based on pair matching. The algorithm has four main parts: 1. Image processing and Feature extraction, 2. Angular distance calculations, 3. Mapping to catalog 4. Attitude determination

#### 1. Image Processing and Feature extraction
 The stars in the image are filtered based on: Brightness, Size, Eccentricity.
 Then we select the N brightest stars in our image with blob detection and the cv2 python module. From this we get the (x, y) coordinates of the center of each star in the image.

 Notes and Ideas: 
 - This can and probably should be optimised using a feature extraction algorithm
 - We currently don't account for lense distortion

#### 2. Angular distance calculations
 - We center the (x, y) coordinates of the stars based on our image's center. After which we convert them into vectors with the focal length being the coordinate by the z axis.
 - Convert them into unit vectors to get rid of unnecessary computation
 - Calculate the angular distance between each pair (the cosine of the dot product of the two vectors in the pair)

#### 3. Mapping stars to catalog (To-do)
  - For each pair of stars in our image we query our database to get the possible candidate pairs of stars with an angular distance within a certain tolerance
  - Now for each star in our image both stars in each candidate pair from the db could be the star in our image. How do we figure out the correct mapping? (TO-DO)
  
  Notes and Ideas:
  - the db search could be seriously optimized using a k-vector
  - my idea for the mapping part is to create a graph-like structure where we have the main nodes be the stars from our image and then their adjacency list could be all the candidate stars from the db. After we generate the graph/dict with such structure we could do a traversal algorithm like DFS/BFS excluding an unlikely pair with each recursion/iteration. If this works the final optimisation woud be to do an A* algorithm if I think of a good heuristic.

### Attitude Determination (To-do)

- From the final mapping of each star to a certain hip in our db which we'll get from the step above we can now get the unit vectors for each hip in **Table 1** of our db 
- Determine pitch and yaw based on the known catalog orientation (aligned with the celestial equator)
- Build transformation matrices from:
  - Image star vectors (camera frame)
  - Catalog star vectors (inertial frame)
- Compute the rotational transformation matrix to determine roll
- Convert the final orientation to **quaternions**

---

## Hardware

### Hardware Integration (To-do)

- The initial orientation determined via camera is stored as a reference
- The IMU is zeroed at this orientation
- Subsequent orientation tracking is performed using only the IMU untill it has to be recalibrated again because of the drift

Notes and ideas:
- The simple way to recalibrate would be to just do it after a certain amount of time has passed. However to produce a more accurate orientation in the end we should look into things like Kalman filtering
---

### Mechanical Design

- Modular, layered physical design:
  - Raspberry Pi
  - IMU module
  - Camera mounted perpendicular to the Raspberry Pi board
- 3D CAD models and assembly diagrams are available in the `DOCS/` directory
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

