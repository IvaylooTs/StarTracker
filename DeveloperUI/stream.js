const streamUrl = "https://192.168.55.160:5000/video_feed";
const img = document.getElementById("stream");
const retryInterval = 5000; // in milliseconds

function tryLoadStream() {
    const testImg = new Image();

    testImg.onload = function () {
        console.log("Stream connected.");
        img.src = streamUrl;
    };

    testImg.onerror = function () {
        console.warn("Stream not available. Retrying in " + retryInterval / 1000 + " seconds...");
        setTimeout(tryLoadStream, retryInterval);
    };

    // Force reload to avoid cached success/failure
    testImg.src = streamUrl + "?t=" + new Date().getTime();
}


window.onload = function () {
    tryLoadStream();
};