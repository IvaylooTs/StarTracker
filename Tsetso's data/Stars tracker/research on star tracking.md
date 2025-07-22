
https://satsearch.co/products/redwirespace-star-tracker

this states that:
- use quarternions for rotation measurement
- 50-60 main navigation star to determent the satellite position. this are probably big and bright stars, simple constellations, not repeated anywhere
- each picture should have at least 7-10 stars?
- pov of ~19x14 degress
- star catalog is 1889 stars

https://www.ty-space.net/what-is-a-star-tracker-and-how-does-it-work-on-satellites/

used for navigation and exploration, use software to determent position

https://www.mouser.bg/ProductDetail/Raspberry-Pi/SC1174?qs=IKkN%2F947nfBRnuXetv8aQQ%3D%3D
it looks like if this is the sensor, we can implement AI into the project, but I think only one person knows how to develop AI in the team, so Deterministic approach (simple algorithms) will be used.



## idea 1
Using photo resistors (basically sensor that detects light) we can use to see where is the sun and with this information change that orientation to point the camera in other directions, this can be used to estimate general direction of the satellite by having direction via IMU (gyroscope, or a.k.a. angle/rotation knowadge) we can estimate which half of the planet we are on.

Pros:
- simple sensors
- much simple algorithms
Cons:
- basically problems with sun
- no precision (like at all! what do you know? if it is day or night???!?!)
- no on task! maybe can be "supportive sub-sub-system", but that is all. not even useful when other system is active!


### Dark frame
This is frame that is captured while the camera looks at the void of my life or just closed with cap or other dark material. This will catch any problem with camera (e.g. noise) and will be subtracted later to create better, darker image, so that small dots and particles (that only exist because the camera is faulty) are removed from the imaged, before processing

[Dark Frame Computation - page 5](https://ntrs.nasa.gov/api/citations/20200001376/downloads/20200001376.pdf?utm_source=chatgpt.com)


### Some info on star position
You also have a database of stars with known **3D positions in the sky**, expressed as:
- Right Ascension (RA) and Declination (Dec), or
- Unit vectors in some inertial frame (e.g., J2000 coordinate system).

### Problem to solve the orientation - Wahba's problem
Popular algorithms to solve this include:
- **QUEST (QUaternion ESTimator)**
- **SVD-based methods**
- **Davenport’s Q-method**



### Will stars position change if i go to the other side of earth

This is called **annual parallax** when Earth orbits the Sun, and it's used to **measure distances to nearby stars**.

But for most spacecraft applications (especially in Earth orbit), the shift is:
- Less than an **arcsecond** (1/3600 of a degree)
- So small that it's usually **negligible for star tracking**
- Unless you're doing **high-precision astrometry** (like Gaia or Hubble)


### star catalog
using some star catalogs can be very useful and save time on mapping. 
[Tycho-2](https://cdsarc.cds.unistra.fr/viz-bin/cat/I/259#/browse) - too big?
Hipparcos - 




### OK here is the idea
1. We get a image of the stars in the sky. 
2. We find which stars are they.
3. We get their position (a.k.a rotation from earth)
4. we know where we are looking at