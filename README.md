# StarTracker
Team Members:Michael Yan, Ivaylo Tsekov, Alexandra Nedela, Dilyana Vasileva, Tsvetomir Staykov
Goal: Create a star tracking system which uses a Raspberry AI camera to determine the orientation of the system based on the star image it receives which works in tandem with an IMU post calibration to keep track of the orientation without live imaging.

Current Design Decisions:
Catalogue processing:
Filter (brightness, size, eccentricity, etc) data to obtain optimal amount of stars for pattern recognition and their cartesian coordinates(database table 1)
create database table of all combination of two stars and the angle between them (database table 2)

Image processing:
filter (brightness, size, eccentricity, etc) image to find 3-20 stars 
Convert pixel coordinates -> camera coordinates -> cartesian space unit vectors of each star
create table of all angles of pairs of found stars. (table 3)

Attitude determination:
search ID of pairs of stars in table 2 with known angles of table 3
find pitch and yaw of camera by matching catalogue orientation (always horizontal along equator)
create two matrices of cartesian positions of image stars and database stars
find rotational transformation matrix between two sets of matrices to find roll of camera.
convert to quaterions.

Hardware:
get orientation of device from camera and program and use it as 'offset'
zero IMU at this point
sum IMU orientation with offset for further movement without camera.

CAD design:
layered design of raspberry Pi and IMU
perpendicular attachement of camera.
image can be seen in DOCS

How to run demo:
create a dark space using planes available.
point device at an image on a screen with known orientation
compare with computed orientation
move device to keep track of orientation without camera

