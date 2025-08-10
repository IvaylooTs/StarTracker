// Stellarium 360-degree rotation script with screenshots
// Takes a screenshot every 45 degrees and names files by rotation angle

// Configuration parameters
var screenshotInterval = 45; // degrees between screenshots
var rotationSpeed = 5; // degrees per step (smaller = smoother)
var stepDelay = 0.1; // seconds between steps
var baseAltitude = 30; // viewing altitude in degrees
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

// Alternative: High-quality rotation with longer pauses
function highQualityRotationScreenshots() {
    core.output("Starting high-quality rotation with extended pauses for screenshots...");
    
    var degrees = [0, 45, 90, 135, 180, 225, 270, 315];
    screenshotCounter = 0;
    
    for (var i = 0; i < degrees.length; i++) {
        var targetDegree = degrees[i];
        
        core.output("Moving to " + targetDegree + " degrees...");
        core.moveToAltAzi(baseAltitude, targetDegree, 2.0);
        core.wait(3.0); // Longer wait for smooth movement and stabilization
        
        // Take screenshot
        takeScreenshot(targetDegree);
        
        // Brief pause between positions
        core.wait(1.0);
    }
    
    core.output("High-quality rotation completed! " + screenshotCounter + " screenshots taken.");
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

// Function to capture specific cardinal and intercardinal directions
function cardinalDirectionsScreenshots() {
    core.output("Capturing screenshots of cardinal and intercardinal directions...");
    
    var directions = [
        {name: "North", degree: 0},
        {name: "Northeast", degree: 45},
        {name: "East", degree: 90},
        {name: "Southeast", degree: 135},
        {name: "South", degree: 180},
        {name: "Southwest", degree: 225},
        {name: "West", degree: 270},
        {name: "Northwest", degree: 315}
    ];
    
    screenshotCounter = 0;
    
    for (var i = 0; i < directions.length; i++) {
        var dir = directions[i];
        
        core.output("Capturing " + dir.name + " (" + dir.degree + "°)...");
        core.moveToAltAzi(baseAltitude, dir.degree, 2.0);
        core.wait(2.5);
        
        // Use direction name in filename
        var filename = "direction_" + dir.degree.toString().padStart(3, '0') + "_" + dir.name.toLowerCase();
        
        try {
            core.screenshot(filename, false, screenshotPath);
            core.output("Screenshot saved: " + filename + ".png");
            screenshotCounter++;
        } catch (e) {
            core.output("Error taking screenshot for " + dir.name + ": " + e.message);
        }
        
        core.wait(1.0);
    }
    
    core.output("Cardinal directions capture completed! " + screenshotCounter + " screenshots taken.");
}

// Function to set custom parameters
function setRotationParameters(newInterval, newAltitude, newStepDelay) {
    screenshotInterval = newInterval || 45;
    baseAltitude = newAltitude || 30;
    stepDelay = newStepDelay || 0.1;
    
    core.output("Parameters updated:");
    core.output("- Screenshot interval: " + screenshotInterval + " degrees");
    core.output("- Viewing altitude: " + baseAltitude + " degrees"); 
    core.output("- Step delay: " + stepDelay + " seconds");
}

// Function to display current settings
function showCurrentSettings() {
    core.output("Current rotation settings:");
    core.output("- Screenshot interval: " + screenshotInterval + " degrees");
    core.output("- Rotation speed: " + rotationSpeed + " degrees per step");
    core.output("- Step delay: " + stepDelay + " seconds");
    core.output("- Viewing altitude: " + baseAltitude + " degrees");
    core.output("- Screenshot path: " + (screenshotPath || "Default Stellarium directory"));
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

// Show current settings
showCurrentSettings();

// Prepare the view
prepareView();

// Option 1: Standard 45-degree interval rotation (DEFAULT)
rotate360WithScreenshots();

// Option 2: High-quality rotation with longer pauses (uncomment to use)
// highQualityRotationScreenshots();

// Option 3: Custom interval - every 30 degrees (uncomment to use)
// customIntervalRotation(30);

// Option 4: Cardinal directions only (uncomment to use)
// cardinalDirectionsScreenshots();

// Option 5: Custom settings then rotation (uncomment to use)
// setRotationParameters(60, 45, 0.2); // 60° intervals, 45° altitude, 0.2s delay
// rotate360WithScreenshots();