	 1. raw images undergo photometric and geometric calibration
 1. image processing algorithms find candidate stars in the calibrated image
 2. image pixel coordinates for each star centroid is converted to a bearing direction
 3. star identification (ID) algorithms match candidate starbearings to a catalog of known star bearings, and
 4.  pairs of corresponding measured/catalog star bearings are used to compute attitude via a solution to Wahba’s problem. These five tasks are 