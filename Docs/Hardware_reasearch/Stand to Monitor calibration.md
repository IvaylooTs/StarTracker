To calibrate the stand for testing the camera with a picture from the screen you can use the pluses/crosses on the  live feed. They are present only on the visualization of the camera from the Flask server, but not in the final image that is processed. Another link to the Flask server path to file `/calibration_overlay` can be use to display on the monitor the calibration image. it will help you match the camera as close as possible to what it should see, if we have direct view toward stars in space. Use program like Stellarium to test your images. Use "F11" to make the browser in full screen and display the image as big as possible.
![[Pasted image 20250816201824.png]]
![[Pasted image 20250816202019.png]]

For the current software version this are always displayed during the live feed. In future 
upgrades they will be toggleable 
### Settings in stellarium !!1! ! seeeee if this exists in the documentation already in different form or something
Use "F3" to rotate to specific object if you want to observe or at specific angle.
Press 'F4" in Stellarium to change the view settings to match the camera. 
The current settings are:
- Projection: Perspective
- Current FOV: 52 degrees 00 minutes 00 seconds
- Labels and Markers: unchecked
- You can remove stars and planet with "P" button. 
- You can remove meteor showers with "CRTL + SHIFT + M".
- Remove HUD with "CTRL + T"
- Remove ground with "G"
- Remove Atmosphere with "A"


Picture with default settings:
![[Pasted image 20250816202409.png]]
Picture with changed settings:
![[Pasted image 20250816202440.png]]