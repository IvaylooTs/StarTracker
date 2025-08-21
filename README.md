# Repo: Star Tracker

**Team Name:** CASSI  
**Project Name:** CASSI (Celestial Alignment System for Satellite Inertial-control)  
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
- `test_images` – dir with images from *Stellarium* for testing purposes (uses specific resolution and FOV)
- `create_hash.py` - a script to generate a hash map from database table "AngularDistances"
- `catalog_hash.py` - an already generated hash with the script above using 2 degree tolerance
- `stellarium360_new.js` - a script to be pasted into stellarium console for testing purposes
- `script_images` - dir with images generated from the above script

To run the script, you will need the `sqlite3` and `cv2` libraries installed on your system, and Python 3.

To run the script, use:
```bash
python3 main.py 
```

The results in your terminal should be:

- Indicator of which image you are currently processing
- Which mode you're in (Lost-in-Space / Tracking)
- The resulting quaternion from either of the two modes if one is produced
- The mapping of image star to HIP ID if one is produced

Files that should be generated when you run the script:

- For each image that you're processing you should get a file generated in this format "stars_identified_[index]"

---

## System Architecture

![Lost-in-Space algorithm Diagram](https://raw.githubusercontent.com/IvaylooTs/StarTracker/refs/heads/main/Docs/Diagrams/Software_diagrams/Lost_in_space.png)

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

---

#### 2. Angular Distance Calculations

- We center the (x, y) coordinates of the stars based on our image's center, then convert them into vectors with the focal length as the coordinate along the z-axis  
- Convert them into unit vectors to reduce unnecessary computation  
- Calculate the angular distance between each pair (using the cosine of the dot product of the two vectors)

---

#### 3. Mapping Stars to Catalog 

- For each pair of stars in our image, we get the candidates ID pairs from our catalog hash
- Then we assign the candidate IDs for each image star based on a voting system
- Afterwards we run a DFS on this graph-like structure to find all possible mappings (even if incomplete)
- In the end we score the solutions we got from the DFS to get the best one

---

#### 4. Attitude Determination 

- From the final mapping of each star to a certain HIP in our database (from step above), we can now retrieve unit vectors from **Table 1**  
- We build the attitude profile matrix from the two sets of image and catalog vectors
- We then build the attitude profile matrix in quaternion form 
- We solve for the eigenvector with the largest eigenvalue which corresponds to our orientation quaternion

---

#### Tracking algorithm

- For tracking we use a previous orientation quaternion to predict where the stars in our previous image are going to be in the next one 
based on a certain amount of euclidian distance threshold this way we can make small and fast corrections to. We use the hungarian algorithm

---

### Hardware

#### Hardware Integration

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
- [x] Identify the correct stars  
- [x] Orientation in quaternion form
- [x] Testing  

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
- [x] Satellite enclosure with removable modules  
  - [x] Bottom panel  
  - [x] Sliders  
  - [x] Test sliders  
  - [x] Side protection panels  
  - [x] CASSI module  

### Additional

- [x] Testing | Make black box for testing using images on a monitor  
- [x] Visualization | Make control panel  
  - [x] 3D visualization  
  - [x] Live video  
  - [x] OpenCV calculations  
  - [x] Control buttons  
- [x] Check if static IP is needed (just in case)  
- [x] Study commercial star tracker manuals  
- [x] Consider handling edge cases such as solar flares, moonlight, sun/earth stray light, etc.
