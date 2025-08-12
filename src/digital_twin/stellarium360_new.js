// 

// Stellarium Script: Go to 0,0 and then rotate
// Moves the view to RA/Dec 0,0, waits for 5 seconds,
// and then demonstrates the available method for turning the view.

// 1. Move to celestial coordinates 0,0
core.moveToRaDec("0", "0");

// 2. Wait for 5 seconds
core.wait(15);

// 3. Rotate the view
// NOTE: A direct rotation to a specific angle (e.g., 45 degrees) is not
// directly supported by the scripting engine. The following commands
// initiate a continuous turn. We can simulate a turn by a rough
// amount by turning for a short duration.

core.moveToRaDec("0", "2");