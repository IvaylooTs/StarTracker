
function displayWebCam() {
    const container = document.getElementById("video-box");
    
    container.style.display = (container.style.display === "none") ? "block" : "none";
    
    let img = document.getElementById("stream")
    
    
    // if (container.style.display === "none") {
        // Disable → remove the image
        // container.removeChild(img);
    if(!img){
        // Enable → create and add the image
        img = document.createElement("img");
        img.id = "stream";
        img.src = "https://192.168.55.160:5000/video_feed";
        img.alt = "Live Stream";
        img.style.width = "100%";
        container.appendChild(img);
    }

}

function displayCLI(){
     const container = document.getElementById("websocket-communication-container");
    
    container.style.display = (container.style.display === "none") ? "block" : "none";
    
    
}