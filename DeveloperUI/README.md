# Cassi Developer Interface

Control different aspects of the Cassi system. Visualize data and check if everything is correct.

#### Current windows:
 - 3D visualization of the quaternions
  - toggleable vectors and objects
 - graph for checking for errors in quaternions.
 - buttons with macro commands for fast execution on important commands - lost in space, calibrate, add offset, compare offsets, shutdown, reboot
 - console for CLI interface and live data transfer
 - camera overview
 - simple display information - current quaternion, temp, ram, cpu usage


#### How to run:
Activate `python3 start.py` to boot into server. Originally we wanted to not use server, but due to Cross-Origin Request problems, simplest solution was to boot python server.
