// Stellarium 360-degree Panoramic Script (Horizontal and Vertical)
// Takes screenshots at specified degree intervals for both azimuth and altitude.

// ————————————————
// CONFIGURATION
// ————————————————

// --- General Settings ---
var rotationSpeed = 5;      // Degrees per step (smaller is smoother, but slower)
var stepDelay = 0.1;        // Seconds between each small step
var screenshotPath = "";    // Leave empty for Stellarium's default screenshots folder

// --- Horizontal Rotation (Azimuth) ---
var horizontalInterval = 45; // Degrees between each horizontal screenshot
var startAzimuth = 0;       // Starting horizontal angle (0 is North)
var endAzimuth = 360;       // Ending horizontal angle
var baseAltitude = 0;       // The vertical angle to maintain during horizontal rotation

// --- Vertical Rotation (Altitude) ---
var verticalInterval = 30;  // Degrees between each vertical screenshot
var startAltitude = 0;      // Starting vertical angle (0 is the horizon)
var endAltitude = 90;       // Ending vertical angle (90 is the zenith)
var fixedAzimuth = 0;       // The horizontal direction to face during vertical rotation (0 is North)


// --- Script Internals (No need to edit below this line) ---
var screenshotCounter = 0;


// ————————————————
// CORE FUNCTIONS
// ————————————————

/**
 * Takes a screenshot with a descriptive filename.
 * @param {string} prefix - The prefix for the filename (e.g., "horizontal" or "vertical").
 * @param {number} degrees - The current angle (azimuth or altitude) for the filename.
 */
function takeScreenshot(prefix, degrees) {
    var filename = prefix + "_" + degrees.toString().padStart(3, '0') + "_deg";
    
    try {
        // Stellarium automatically adds the .png extension
        core.screenshot(filename, false, screenshotPath);
        core.output("Screenshot saved: " + filename + ".png (at " + degrees + " degrees)");
        screenshotCounter++;
    } catch (e) {
        core.output("Error taking screenshot at " + degrees + " degrees: " + e.message);
    }
}

/**
 * Prepares the view by setting a default position and waiting for it to stabilize.
 */
function prepareView() {
    core.output("Preparing view for rotation...");
    // Reset view to the starting position for horizontal rotation
    core.moveToAltAzi(baseAltitude, startAzimuth, 1.0);
    core.wait(1.5); // Wait for view to stabilize
    core.output("View is ready.");
}

/**
 * Performs a full horizontal rotation (azimuth) while taking screenshots.
 */
function rotateHorizontallyWithScreenshots() {
    core.output("Starting HORIZONTAL rotation, taking screenshots every " + horizontalInterval + " degrees...");
    
    screenshotCounter = 0;
    var currentDegree = startAzimuth;
    var nextScreenshotAt = startAzimuth;
    
    // Take initial screenshot at the starting position
    core.moveToAltAzi(baseAltitude, startAzimuth, 1.0);
    core.wait(1.0);
    takeScreenshot("horizontal", nextScreenshotAt);
    nextScreenshotAt += horizontalInterval;
    
    // Main horizontal rotation loop
    while (currentDegree < endAzimuth) {
        currentDegree += rotationSpeed;
        
        var azimuth = currentDegree % 360;
        core.moveToAltAzi(baseAltitude, azimuth, 0.3);
        core.wait(stepDelay);
        
        // Check if it's time for a screenshot
        if (currentDegree >= nextScreenshotAt && nextScreenshotAt < endAzimuth) {
            core.wait(0.7); // Extra wait for stabilization before screenshot
            takeScreenshot("horizontal", nextScreenshotAt);
            nextScreenshotAt += horizontalInterval;
        }
        
        // Progress indicator
        if (currentDegree % 90 === 0 && currentDegree > 0) {
            core.output("Horizontal Progress: " + currentDegree + "° completed");
        }
    }
    
    core.output("Horizontal rotation complete! Took " + screenshotCounter + " screenshots.");
}


/**
 * Performs a vertical rotation (altitude) while taking screenshots.
 */
function rotateVerticallyWithScreenshots() {
    core.output("Starting VERTICAL rotation, taking screenshots every " + verticalInterval + " degrees...");
    
    screenshotCounter = 0;
    var currentAltitude = startAltitude;
    var nextScreenshotAt = startAltitude;

    // Move to the starting position
    core.moveToAltAzi(startAltitude, fixedAzimuth, 1.0);
    core.wait(1.5); // Wait for stabilization
    takeScreenshot("vertical", nextScreenshotAt);
    nextScreenshotAt += verticalInterval;

    // Main vertical rotation loop
    while (currentAltitude < endAltitude) {
        currentAltitude += rotationSpeed;

        // Move to the new altitude
        core.moveToAltAzi(currentAltitude, fixedAzimuth, 0.3);
        core.wait(stepDelay);

        // Check if it's time for a screenshot
        if (currentAltitude >= nextScreenshotAt && nextScreenshotAt <= endAltitude) {
            core.wait(0.7); // Extra wait for stabilization
            takeScreenshot("vertical", nextScreenshotAt);
            nextScreenshotAt += verticalInterval;
        }
    }

    core.output("Vertical rotation complete! Took " + screenshotCounter + " screenshots.");
}


// ————————————————
// SCRIPT EXECUTION
// ————————————————

// Prepare the view before starting any rotation.
prepareView();

// --- CHOOSE WHICH ROTATION TO RUN ---
// You can run one, the other, or both. Just uncomment the functions you want to execute.

// Option 1: Run the horizontal rotation
rotateHorizontallyWithScreenshots();

// Option 2: Run the vertical rotation
// rotateVerticallyWithScreenshots();

core.output("All selected script actions have been completed.");