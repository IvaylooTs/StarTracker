# StarTracker

**Team Members:**  
Michael Yan, Ivaylo Tsekov, Alexandra Nedela, Dilyana Vasileva, Tsvetomir Staykov

---

## Project Overview

This project implements a star tracking system using a Raspberry Pi AI camera to determine the orientation of the system based on captured star images. Once calibrated, the system works in tandem with an IMU (Inertial Measurement Unit) to track orientation without continuous visual input.

---

## System Architecture

### Catalogue Processing

- Filter stars based on:
  - Brightness
  - Size
  - Eccentricity
- Select the optimal set of stars for pattern recognition
- Convert selected stars to Cartesian coordinates (stored in **Database Table 1**)
- Generate all possible star pairs and compute angles between them (stored in **Database Table 2**)

### Image Processing

- Filter stars in the image based on:
  - Brightness
  - Size
  - Eccentricity
- Detect approximately 3–20 stars from the image
- Convert pixel coordinates to camera coordinates, then to Cartesian unit vectors
- Calculate angles between all star pairs (stored in **Table 3**)

### Attitude Determination

- Match angles from **Table 3** with catalogue data in **Table 2** to identify observed stars
- Determine pitch and yaw based on the known catalogue orientation (aligned with the celestial equator)
- Build transformation matrices from:
  - Image star vectors (camera frame)
  - Catalogue star vectors (inertial frame)
- Compute the rotational transformation matrix to determine roll
- Convert the final orientation to **quaternions**

---

## Hardware Integration

- The initial orientation determined via camera is stored as a reference
- The IMU is zeroed at this orientation
- Subsequent orientation tracking is performed using only the IMU

**Final Orientation = IMU Output + Initial Camera-Based Offset**

---

## Mechanical Design

- Modular, layered physical design:
  - Raspberry Pi
  - IMU module
  - Camera mounted perpendicular to the Raspberry Pi board
- 3D CAD models and assembly diagrams are available in the `DOCS/` directory

---

## Demonstration Instructions

1. Set up a dark environment using panels or an enclosure.
2. Point the device at a star image displayed on a screen with a known orientation.
3. Verify that the computed orientation matches the actual orientation.
4. Move the device and observe that orientation tracking continues using the IMU, even without camera input.

---
