# Repo: Star Tracker

**Team Name:** 404: Star Not Found  
**Unofficial Project Name:** CASSI (Celestial Alignment System for Satellite Inertial-control)  
**Official Category:** Star Tracker 1

**Team Members:**  
Michael Yan, Ivaylo Tsekov, Alexandra Nedela, Dilyana Vasileva, Tsvetomir Staykov

---

## Problem

How do we determine our attitude in space?

---

## Project Overview and End Goal

This project implements a star tracking system using a **Raspberry Pi** with an **AI camera** to determine the orientation of the system based on captured star images using a deterministic algorithm. Once calibrated, the system works in tandem with an IMU (Inertial Measurement Unit) to track orientation without continuous visual input. When drifting occurs, due to environmental variables, power loss, or other factors, the system automatically recalibrates.

---

## Current Available Demo

Go to the `digital_twin` folder in the `src` directory. It contains:

- `main.py` – main file with all the systems  
- `star_distances_sorted.db` – database with calculated angular distances, sorted in ascending order  
- `star_identified.png` – sample image from previous tests  
- `test_images` – good images from *Stellarium* for testing purposes (uses specific resolution and FOV)

To run the script, you will need the `sqlite3` and `cv2` libraries installed on your system, and Python 3.

To run the script, use:
```bash
python3 main.py 
```

The results in your terminal should be:

- Pixel coordinates of stars found in the picture  
- The angular distance of each pair (close to the real ones — needs optimization)  
- Candidate matches from the database for each star in the image  
- The image being tested should display with the N brightest stars circled  

---

## System Architecture

![Algorithm Diagram](https://raw.githubusercontent.com/IvaylooTs/StarTracker/refs/heads/main/Docs/Diagrams/system.png)

![Cad Design](https://raw.githubusercontent.com/IvaylooTs/StarTracker/refs/heads/main/Docs/CAD%20Images%20%2B%20drawings/CAD%20image.png)

### Software

#### Database Structure

We have two tables in our database.  
**Table 1 – "Stars"** contains the HIP of each star according to the Hipparcos catalog, the magnitude of each star, and x, y, z coordinates of a unit vector corresponding to their RA and DEC coordinates in the equatorial coordinate system.  
**Table 2 – "AngularDistances"** contains the calculated angular distance between each pair of stars in Table 1, sorted from lowest to highest value.

The stars are filtered based on:

- Brightness  
- Size  
- Eccentricity

#### Lost-in-Space Algorithm

We're implementing a Lost-in-Space approach based on pair matching. The algorithm has four main parts:  
1. Image Processing and Feature Extraction  
2. Angular Distance Calculations  
3. Mapping to Catalog  
4. Attitude Determination

---

#### 1. Image Processing and Feature Extraction

The stars in the image are filtered based on: Brightness, Size, and Eccentricity.  
Then we select the N brightest stars in our image using blob detection and the `cv2` Python module. From this, we get the (x, y) coordinates of the center of each star in the image.

Notes and Ideas:

- This can and probably should be optimized using a feature extraction algorithm  
- We currently don't account for lens distortion  

---

#### 2. Angular Distance Calculations

- We center the (x, y) coordinates of the stars based on our image's center, then convert them into vectors with the focal length as the coordinate along the z-axis  
- Convert them into unit vectors to reduce unnecessary computation  
- Calculate the angular distance between each pair (using the cosine of the dot product of the two vectors)

---

#### 3. Mapping Stars to Catalog (To-do)

- For each pair of stars in our image, we query our database to get the possible candidate pairs with angular distances within a certain tolerance  
- For each star in our image, both stars in each candidate pair from the database could be a match. How do we figure out the correct mapping? (TO-DO)

Notes and Ideas:

- The database search could be seriously optimized using a k-vector  
- One idea for the mapping part is to create a graph-like structure, where the main nodes are the stars from our image, and their adjacency lists contain all the candidate stars from the database. After generating this graph/dictionary, we could use a traversal algorithm like DFS/BFS, excluding unlikely pairs at each recursion/iteration. If this works, the final optimization would be to use an A* algorithm with a good heuristic.

---

#### 4. Attitude Determination (To-do)

- From the final mapping of each star to a certain HIP in our database (from step above), we can now retrieve unit vectors from **Table 1**  
- Determine pitch and yaw based on the known catalog orientation (aligned with the celestial equator)  
- Build transformation matrices from:  
  - Image star vectors (camera frame)  
  - Catalog star vectors (inertial frame)  
- Compute the rotational transformation matrix to determine roll  
- Convert the final orientation to **quaternions**

---

### Hardware

#### Hardware Integration (To-do)

- The initial orientation determined via camera is stored as a reference  
- The IMU is zeroed at this orientation  
- Subsequent orientation tracking is performed using only the IMU until recalibration is required due to drift

Notes and Ideas:
- A simple way to recalibrate would be to do it periodically  
- For improved accuracy, consider using Kalman filtering

---

#### Mechanical Design

- Modular, layered physical design:
  - Raspberry Pi  
  - IMU module  
  - Camera mounted perpendicular to the Raspberry Pi board  

- 3D CAD models and assembly diagrams are available in the `DOCS/` directory

---

##  Current Progress

### Software

- [x] Define project requirements  
- [x] Find algorithm for identification of stars  
- [x] OpenCV star identification on image  
  - [x] Test model with math on hand  
- [x] Get angular distances between stars  
- [x] Create database of parameters for comparison (star 1 HIP, star 2 HIP, angular distance)  
- [x] Select candidates for possible stars in specific locations  
- [ ] Identify the correct stars  
- [ ] Find rotation matrix  
- [ ] Apply rotation vector to IMU measurements  
- [ ] Testing  

### Hardware

- [x] Configure Raspberry Pi  
- [x] Connect required modules  
  - [x] Camera  
  - [x] IMU (Gyroscope, Accelerometer, Magnetometer)  
- [x] Calibrate IMU at hardware level  
  - [ ] Measure drift  
  - [ ] Research drift correction  
- [ ] Calibrate IMU at software level  
- [ ] Add interface for communication with Arduino (I2C is a good candidate)  
- [ ] Develop firmware  
  - [ ] Auto boot configuration  

### CAD

- [x] Construct simple box for fit testing and basic implementation  
- [x] Build a better box for testing and protection  
- [ ] Satellite enclosure with removable modules  
  - [ ] Bottom panel  
  - [ ] Sliders  
  - [ ] Test sliders  
  - [ ] Side protection panels  
  - [ ] CASSI module  

### Additional

- [ ] Testing | Make black box for testing using images on a monitor  
- [ ] Visualization | Make control panel  
  - [ ] 3D visualization  
  - [ ] Live video  
  - [ ] OpenCV calculations  
  - [ ] Control buttons  
- [ ] Check if static IP is needed (just in case)  
- [ ] Study commercial star tracker manuals  
- [ ] Consider handling edge cases such as solar flares, moonlight, sun/earth stray light, etc.
