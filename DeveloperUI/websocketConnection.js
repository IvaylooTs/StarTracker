
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

function quatRotateVec(q, v) {
    const [w, x, y, z] = q;
    const vx = 2*(y*v[2] - z*v[1]);
    const vy = 2*(z*v[0] - x*v[2]);
    const vz = 2*(x*v[1] - y*v[0]);
    return [
        v[0] + w*vx + (y*vz - z*vy),
        v[1] + w*vy + (z*vx - x*vz),
        v[2] + w*vz + (x*vy - y*vx)
    ];
}

function axisAngleDeg(q1, q2, axis = [1,0,0]) {
    const a1 = quatRotateVec(q1, axis);
    const a2 = quatRotateVec(q2, axis);
    const dot = a1[0]*a2[0] + a1[1]*a2[1] + a1[2]*a2[2];
    const len1 = Math.hypot(...a1), len2 = Math.hypot(...a2);
    return Math.acos(Math.max(-1, Math.min(1, dot/(len1*len2)))) * 180/Math.PI;
}

// Example:




function ConnectToWebSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        return; // Already connected or in progress
    }

    try {
        ws = new WebSocket('ws://192.168.55.160:6789');

        let timeout = setTimeout(() => {
            if (ws.readyState !== WebSocket.OPEN) {
                ws.close();
                TerminalDisplayInfo("Connection timed out.");
            }
        }, 2500); // 5 sec timeout

        ws.onopen = () => {
            clearTimeout(timeout);
            console.log('WebSocket connected');
            TerminalDisplayInfo("Connected to server.");
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

                    let c_x = parseFloat(current.x);
                    let c_y = parseFloat(current.y);
                    let c_z = parseFloat(current.z);
                    let c_w = parseFloat(current.w);
                    
                    console.log(c_x,c_y,c_z,c_w);
                    // window.Arrow3DObject.quaternion.set(current.x,current.y,current.z, current.w);
                    
                    
                    window.Arrow3DObjectCalibration.quaternion.set(c_x, c_y,c_z,c_w);
                    let o_x = parseFloat(old.x);
                    let o_y = parseFloat(old.y);
                    let o_z = parseFloat(old.z);
                    let o_w = parseFloat(old.w);
                    window.Arrow3DObjectCalibrationOlder.quaternion.set(o_x, o_y,o_z,o_w);

                    
                    let angle = axisAngleDeg([o_w, o_x, o_y, o_z], [ c_w, c_x, c_y, c_z]); // ~90
                    CLIDisplayInfo("Angle between them: "
                        +angle);
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
            clearTimeout(timeout);
            TerminalDisplayInfo("Connection lost. Retrying...");
            removeCameraElement();
            retryWebSocketConnection();
        };

        ws.onerror = (err) => {
            clearTimeout(timeout);
            console.error('WebSocket error:', err);
            TerminalDisplayInfo("WebSocket error.");
            ws.close(); // triggers onclose -> retry
        };

    } catch (e) {
        console.error("Failed to create WebSocket:", e);
        TerminalDisplayInfo("WebSocket creation failed.");
        retryWebSocketConnection();
    }
}

function retryWebSocketConnection() {
    if (connectionRetryInterval) return; // Already scheduled

    connectionRetryInterval = setTimeout(() => {
        connectionRetryInterval = null; // Reset so next failure can retry
        console.log("Attempting to reconnect...");
        ConnectToWebSocket();
    }, 1000); // Retry after 1 second
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
            CLIDisplayInfo("Sending calibrating IMU");
        } else if(cmd === 'addOffset') {
            const w = parseFloat(document.getElementById("q_w").value);
            const x = parseFloat(document.getElementById("q_x").value);
            const y = parseFloat(document.getElementById("q_y").value);
            const z = parseFloat(document.getElementById("q_z").value);

            CLIDisplayInfo("Adding offset of "+w +" "+ x +" " +y + " " +z);
            if ([w, x, y, z].some(isNaN)) {
                displayError("Please enter all quaternion values (w, x, y, z).");
                return;
            }
            ws.send(JSON.stringify({ action: cmd, data: {w, x, y, z} }));
            ws.send(JSON.stringify({ action: "getCalibrationQuaternions" }));

        } else {
            ws.send(JSON.stringify({ action: cmd }));

        }
    } else {
        // CLIDisplayInfo("> " + cmd);
        displayError("WebSocket not connected");
    }

    
}

// window.addEventListener('beforeunload', () => {
//     if (ws && ws.readyState === WebSocket.OPEN) {
//         ws.send(JSON.stringify({ action: "disconnecting" }));
//         ws.close(1000, "Client closed connection");
//     }
// });

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
