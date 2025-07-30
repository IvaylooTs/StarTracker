
function TerminalDisplayInfo(message){
    const messageDiv = document.getElementById('messages');
    const p = document.createElement('p');
    p.textContent = message;
    messageDiv.appendChild(p);
    messageDiv.scrollTop = messageDiv.scrollHeight;
}

const quaternionOutput = document.getElementById('quaternionOutput');
const systemStats = document.getElementById('systemStats');

const ws = new WebSocket('ws://192.168.55.160:6789');

ws.onopen = () => console.log('WebSocket connected');

ws.onmessage = function(event) {
    TerminalDisplayInfo(event.data);    
    try {
    const msg = JSON.parse(event.data);

        if (msg.quaternion) {
        const { x, y, z, w } = msg.quaternion;
        quaternionOutput.textContent = `w: ${w}\nx: ${x}\ny: ${y}\nz: ${z}\n`;
        systemStats.textContent = `CPU Usage: ${msg.CPU_temp}°C\nMemory Usage: ${msg.CPU_usage}%\nRam usage: ${msg.RAM_percent}%`;

        console.log({x,y,z,w});
        window.Satellite3DObject.quaternion.set(x,y,z,w);
        AddDataToChart({ w, x, y, z });
        // window.Satellite3DObject.quaternion.set(0.707,0.707,0,0);
        // window.Satellite3DObject.quaternion.set(  0.966,0,0.259,0);
        }
    } catch (e) {
        console.warn('Invalid or partial JSON:', e);
    }
};

function sendCommand(cmd) {
    if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: cmd }));
        alert("cmd command send!");
        if(cmd === 'calibrate') {
            // shouldAddCalibrationLine =  true; // not used
        }
    } else {
    alert("WebSocket not connected");
    }
}  

ws.onclose = () => {
    console.error('WebSocket error:', err)
    TerminalDisplayInfo("Connection lost.");
};
ws.onerror = (err)  => {
    console.error('WebSocket error:', err);
    TerminalDisplayInfo("Error!");
};


async function downloadPhotoCapture(url = 'https://192.168.55.160:5000/capture_photo', filename = 'capture.jpg') {
    try {
    const response = await fetch(url);
    if (!response.ok) throw new Error('Network response was not OK');
    
    const blob = await response.blob();
    const blobUrl = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = blobUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    
    URL.revokeObjectURL(blobUrl);  // clean up memory
    } catch (error) {
    console.error('Download failed:', error);
    }
}