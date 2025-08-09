
function TerminalDisplayInfo(message){
    const messageDiv = document.getElementById('messages');
    const MAX_MESSAGES = 100;

    const p = document.createElement('p');
    p.textContent = message;
    messageDiv.appendChild(p);

    while (messageDiv.children.length > MAX_MESSAGES) {
        messageDiv.removeChild(messageDiv.firstChild);
    }

    messageDiv.scrollTop = messageDiv.scrollHeight;
}



function CLIDisplayInfo(message){
    const CLImessageDiv = document.getElementById('cli-cmds-contained');
    const MAX_MESSAGES = 100;

    const p = document.createElement('p');
    p.textContent = message;
    CLImessageDiv.appendChild(p);

    while (CLImessageDiv.children.length > MAX_MESSAGES) {
        CLImessageDiv.removeChild(CLImessageDiv.firstChild);
    }

    CLImessageDiv.scrollTop = CLImessageDiv.scrollHeight;
}



const quaternionOutput = document.getElementById('quaternionOutput');
const systemStats = document.getElementById('systemStats');

let ws = null;
let connectionRetryInterval = null;

let firstLogin = false

function removeCameraElement(){
    let img = document.getElementById("stream")
    if(img){
        img.remove();
    }
}


let latestCalibration, olderCalibration;

function ConnectToWebSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        return; // Already connected or trying
    }
    try {
        ws = new WebSocket('ws://192.168.55.160:6789');

        const timeout = setTimeout(() => {
            if (ws.readyState !== WebSocket.OPEN) {
                ws.close();
                TerminalDisplayInfo("Connection timed out.");
            }
        },5000); // 5 seconds
        ws.onopen = () => clearTimeout(timeout);

    } catch (e) {
        console.error("Failed to create WebSocket:", e);
        TerminalDisplayInfo("WebSocket creation failed.");
        return; // Prevent running rest of connect logic
    }

    ws.onopen = () => {
        console.log('WebSocket connected');
        TerminalDisplayInfo("Connected to server.");
        if (connectionRetryInterval) {
            clearInterval(connectionRetryInterval);
            connectionRetryInterval = null;
        }
    };



    ws.onmessage = function(event) {
        TerminalDisplayInfo(event.data);    
        try {
            const msg = JSON.parse(event.data);

            if (msg.quaternion) {
                const { x, y, z, w } = msg.quaternion;
                quaternionOutput.textContent = `w: ${w}\nx: ${x}\ny: ${y}\nz: ${z}\n`;
                systemStats.textContent = `CPU temp: ${msg.stats.CPU_temp}°C CPU Usage: ${msg.statsCPU_usage}%Ram usage: ${msg.stats.RAM_percent}%`;

                // console.log({x, y, z, w});
                window.Satellite3DObject.quaternion.set(x, y, z, w);
                AddDataToChart({ w, x, y, z });
            }
            else if(msg.error){
                CLIDisplayInfo("Received error: " + msg.error.message);
            }else if(msg.ack){
                CLIDisplayInfo("Received ack: " + msg.ack.message);
            }else if(msg.calibrationInfo){
                // CLIDisplayInfo("fuck me ig");
                let current =msg.calibrationInfo.current;
                let old = msg.calibrationInfo.old;
                let currentData = " w: " + current.w + " x: " + current.x + " y: "+ current.y + " z: " + current.z;
                let oldData = "w: " + old.w + " x: " + old.x + " y: "+ old.y + " z: " + old.z;
                latestCalibration = current;
                olderCalibration = old;
                // window.Arrow3DObject.quaternion.set(1,0,0,0)
                console.log(current);
                console.log(old);

                x = parseFloat(current.x);
                y = parseFloat(current.y);
                z = parseFloat(current.z);
                w = parseFloat(current.w);
                
                console.log(x,y,z,w);
                // window.Arrow3DObject.quaternion.set(current.x,current.y,current.z, current.w);
                
                
                window.Arrow3DObjectCalibration.quaternion.set(x, y,z,w);
                x = parseFloat(old.x);
                y = parseFloat(old.y);
                z = parseFloat(old.z);
                w = parseFloat(old.w);
                window.Arrow3DObjectCalibrationOlder.quaternion.set(x, y,z,w);
                // window.Arrow3DObject.setRotationFromQuaternion(
                // new THREE.Quaternion(current.x, current.y, current.z, current.w)
                // );

                // window.setCalibrationArrowQuaternion(current.w,current.x, current.y, current.z);
                // RotateCalibrationArrow(current)
                CLIDisplayInfo("Received calibration [current | old ]: " + currentData + " | " + oldData);
            }

        } catch (e) {
            console.warn('Invalid or partial JSON:', e);
        }
    };

    ws.onclose = () => {
        TerminalDisplayInfo("Connection lost. Retrying...");
        console.warn("WebSocket closed.");
        retryWebSocketConnection();
        removeCameraElement();
    };

    ws.onerror = (err) => {
        console.error('WebSocket error:', err);
        TerminalDisplayInfo("WebSocket error.");
        ws.close();
    };
}

function retryWebSocketConnection() {
    if (connectionRetryInterval) return; // Already retrying

    connectionRetryInterval = setInterval(() => {
        if (!ws || ws.readyState === WebSocket.CLOSED) {
            console.log("Attempting to reconnect...");
            ConnectToWebSocket();
        }
    }, 1000); // Retry every 3 seconds
}


function displayError(message){
    CLIDisplayInfo("# " + message);
    alert(message);
}
function sendCommandCLI(){
    const textBoxInput = document.getElementById('CLI-input');
    sendCommand(textBoxInput.value)
}

function sendCommand(cmd) {
    console.log("sending",cmd);
    CLIDisplayInfo("> " + cmd);
    if (ws.readyState === WebSocket.OPEN) {
        if (!confirm("Are you sure you want to send command: \"" + cmd + "\"?"))
            return;
        

        if(cmd === 'calibrate') {
            ws.send(JSON.stringify({ action: cmd }));
            CLIDisplayInfo("# Sending calibrating IMU");
        } else if(cmd === 'addOffset') {
            const w = parseFloat(document.getElementById("q_w").value);
            const x = parseFloat(document.getElementById("q_x").value);
            const y = parseFloat(document.getElementById("q_y").value);
            const z = parseFloat(document.getElementById("q_z").value);

            CLIDisplayInfo("# Adding offset of "+w +" "+ x +" " +y + " " +z);

            if ([w, x, y, z].some(isNaN)) {
                displayError("# Please enter all quaternion values (w, x, y, z).");
                return;
            }
            ws.send(JSON.stringify({ action: cmd, data: {w, x, y, z} }));
        } else {
            ws.send(JSON.stringify({ action: cmd }));

        }
    } else {
        CLIDisplayInfo("> " + cmd);
        displayError("# WebSocket not connected");
    }

    
}

window.addEventListener('beforeunload', () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: "disconnecting" }));
        ws.close(1000, "Client closed connection");
    }
});

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
        
        URL.revokeObjectURL(blobUrl);
    } catch (error) {
        console.error('Download failed:', error);
    }
}

// Start the connection
ConnectToWebSocket();
