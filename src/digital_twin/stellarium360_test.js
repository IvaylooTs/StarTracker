// Stellarium 360-degree rotation script with screenshots
// Takes a screenshot every 45 degrees and names files by rotation angle

// Configuration parameters
var screenshotInterval = 45; // degrees between screenshots
var rotationSpeed = 5; // degrees per step (smaller = smoother)
var stepDelay = 0.1; // seconds between steps
var baseAltitude = 0; // viewing altitude in degrees
var screenshotPath = ""; // Leave empty for default Stellarium screenshots folder

// Screenshot counter and naming
var screenshotCounter = 0;

// Function to take screenshot with custom name
function takeScreenshot(degrees) {
    var filename = "rotation_" + degrees.toString().padStart(3, '0') + "_deg";
    
    try {
        // Take screenshot - Stellarium will add .png extension automatically
        core.screenshot(filename, false, screenshotPath);
        core.output("Screenshot saved: " + filename + ".png (at " + degrees + " degrees)");
        screenshotCounter++;
    } catch (e) {
        core.output("Error taking screenshot at " + degrees + " degrees: " + e.message);
    }
}

// Main 360-degree rotation with screenshots
function rotate360WithScreenshots() {
    core.output("Starting 360-degree rotation with screenshots every " + screenshotInterval + " degrees...");
    core.output("Screenshots will be saved to Stellarium's default directory");
    
    var totalDegrees = 360;
    var currentDegree = 0;
    var nextScreenshotAt = 0;
    
    // Reset screenshot counter
    screenshotCounter = 0;
    
    // Take initial screenshot at 0 degrees
    core.moveToAltAzi(baseAltitude, 0, 1.0);
    core.wait(1.0); // Wait for view to stabilize
    takeScreenshot(0);
    nextScreenshotAt = screenshotInterval;
    
    // Main rotation loop
    while (currentDegree < totalDegrees) {
        currentDegree += rotationSpeed;
        
        // Move to new position
        var azimuth = currentDegree % 360;
        core.moveToAltAzi(baseAltitude, azimuth, 0.3);
        core.wait(stepDelay);
        
        // Check if it's time for a screenshot
        if (currentDegree >= nextScreenshotAt && nextScreenshotAt < totalDegrees) {
            // Wait a bit longer for the view to stabilize
            core.wait(0.5);
            takeScreenshot(nextScreenshotAt);
            nextScreenshotAt += screenshotInterval;
        }
        
        // Progress indicator every 90 degrees
        if (currentDegree % 90 === 0 && currentDegree > 0) {
            core.output("Progress: " + currentDegree + "° completed");
        }
    }
    
    core.output("Rotation completed! Took " + screenshotCounter + " screenshots total.");
    core.output("Screenshots saved as: rotation_000_deg.png, rotation_045_deg.png, etc.");
}


// Function for custom degree intervals
function customIntervalRotation(intervalDegrees) {
    core.output("Starting rotation with screenshots every " + intervalDegrees + " degrees...");
    
    var totalDegrees = 360;
    var currentPosition = 0;
    screenshotCounter = 0;
    
    while (currentPosition < totalDegrees) {
        core.output("Capturing at " + currentPosition + " degrees...");
        
        // Move to position
        core.moveToAltAzi(baseAltitude, currentPosition, 1.5);
        core.wait(2.0); // Wait for stabilization
        
        // Take screenshot
        takeScreenshot(currentPosition);
        
        // Move to next position
        currentPosition += intervalDegrees;
        
        // Don't exceed 360 degrees
        if (currentPosition >= 360) {
            break;
        }
        
        core.wait(0.5); // Brief pause between captures
    }
    
    core.output("Custom interval rotation completed! " + screenshotCounter + " screenshots taken.");
}


// Utility function to clean up view before starting
function prepareView() {
    core.output("Preparing view for rotation...");
    
    // Reset view to a standard position
    core.moveToAltAzi(baseAltitude, 0, 1.0);
    core.wait(1.0);
    
    // Optional: Set time to a specific value for consistent lighting
    // core.setTimeRate(0); // Pause time
    
    core.output("View prepared. Ready for rotation.");
}

// Main execution options:


// Prepare the view
prepareView();

// Option 1: Standard 45-degree interval rotation (DEFAULT)
rotate360WithScreenshots();

