Step by step algorithm working:
 1. raw images undergo photometric and geometric calibration
 2. image processing algorithms find candidate stars in the calibrated image
 3. image pixel coordinates for each star centroid is converted to a bearing direction
 4. star identification (ID) algorithms match candidate starbearings to a catalog of known star bearings, and
 5.  pairs of corresponding measured/catalog star bearings are used to compute attitude via a solution to Wahba’s problem. These five tasks are 


Webcam information:
general info - https://www.mouser.bg/new/raspberry-pi/raspberry-pi-ai-camera/
data sheet - https://www.mouser.bg/pdfDocs/ai-camera-product-brief.pdf