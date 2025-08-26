
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
        ws = new WebSocket(`ws://${window.ip}:6789`);


        let stateCurrentStats = document.getElementById("stateValue");
        let trustValue = document.getElementById("trustValue");
        let modeCurrentStats =  document.getElementById("modeValue");
        let currentAngleState = document.getElementById("currentAngleState");
        let timeout = setTimeout(() => {
            if (ws.readyState !== WebSocket.OPEN) {
                ws.close();
                TerminalDisplayInfo("Connection timed out.");
            }
        }, 1000); // 5 sec timeout

        ws.onopen = () => {
            clearTimeout(timeout);
            console.log('WebSocket connected');
            TerminalDisplayInfo("Connected to server.");
        };
function multiplyMsgQuaternion(x1,y1,z1,w1) {
  // Destructure the input quaternion components
    x =x1;
     y = y1; z= z1; w= w1; 

  // Define the second quaternion
  let x2 = 0;
  let y2 = 0;
  let z2 = 0.383;
  let w2 = 0.934;

  // Quaternion multiplication
  return {
    w: w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    x: w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
    y: w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
    z: w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
  };
}


        ws.onmessage = function(event) {
            TerminalDisplayInfo(event.data);
            try {
                const msg = JSON.parse(event.data);
                if (msg.quaternion) {
                    let { x, y, z, w } = msg.quaternion;
                    window.Satellite3DObject.quaternion.set(x,y,z,w);
                    AddDataToChart({ w, x, y, z });
                    w =parseFloat(w).toFixed(4);
                    x =parseFloat(x).toFixed(4);
                    y =parseFloat(y).toFixed(4);
                    z =parseFloat(z).toFixed(4);
                    let addMinusW = (w < 0 ? "-" : " ")
                    let addMinusX = (x < 0 ? "-" : " ")
                    let addMinusY = (y < 0 ? "-" : " ")
                    let addMinusZ = (z < 0 ? "-" : " ")
                    w = Math.abs(w)
                    x = Math.abs(x)
                    y = Math.abs(y)
                    z = Math.abs(z)
                    quaternionOutput.textContent = `w: ${addMinusW}${w} | x: ${addMinusX}${x}\ny: ${addMinusY}${y} | z: ${addMinusZ}${z}`;
                    systemStats.textContent = `CPU temp: ${msg.stats.CPU_temp}°C\nCPU Usage: ${msg.stats.CPU_usage}%\nRam usage: ${msg.stats.RAM_percent}%`;

                    let trustFactor = msg.trust;
                    
                    switch(parseInt(trustFactor)){
                        case 0:
                            trustValue.style.color = "#DC143C"
                            trustValue.textContent = "0 - IMU raw"
                            break;
                        case 1:
                            trustValue.style.color = "#FCF55F"
                            trustValue.textContent = "1 - IMU calibrated"
                            break;
                        case 2:
                            trustValue.style.color = "#AFE1AF"
                            trustValue.textContent = "2 - tracking no IMU"
                            break;
                        case 3:
                            trustValue.style.color = "#228B22"
                            trustValue.textContent = "3 - tracking"
                            break;
                        default:
                            trustValue.textContent = msg.trust

                    }

                    let cur_mode = msg.currentMode;
                    let cur_lost_in_space_state = msg.lostInSpaceState;

                    if(cur_mode == "manual"){
                        modeCurrentStats.style.color = 'white';
                        // modeCurrentStats.style.color = 'orange';
                    }else if(cur_mode =="auto"){
                    }
                    currentAngleState.textContent = parseFloat(msg.angleDiff).toFixed(4)+ "°";

                    if(cur_lost_in_space_state.toLowerCase() == "inactive"){
                        stateCurrentStats.style.color = 'white';
                    }else 
                    if(cur_lost_in_space_state.toLowerCase() == "lost in space"){
                        stateCurrentStats.style.color = '#4CBB17';
                    }
                    else if(cur_lost_in_space_state.toLowerCase() == "tracking"){
                        stateCurrentStats.style.color = '';
                    }
                    else if(cur_lost_in_space_state.toLowerCase() == "imu calibrated"){
                        stateCurrentStats.style.color = '#FF5F1F';
                    }
                    else if(cur_lost_in_space_state.toLowerCase() == "imu raw"){
                        stateCurrentStats.style.color = '#DC143C';
                    }

                    modeCurrentStats.textContent  =  cur_mode.toUpperCase();
                    stateCurrentStats.textContent =  cur_lost_in_space_state.toUpperCase();


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
                    let angle = msg.calibrationInfo.angle;

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

                    
                    // let angle = axisAngleDeg([o_w, o_x, o_y, o_z], [ c_w, c_x, c_y, c_z]); // ~90
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

function handleDropdownChange(type) {
    let value;
    if (type === "returnMode") {
        value = document.getElementById("returnMode").value;
        CLIDisplayInfo("Return quaternion mode set to: " + value);
        ws.send(JSON.stringify({ action: "setReturnMode", mode: value }));
    } 
    else if (type === "quatSource") {
        value = document.getElementById("quatSource").value;
        CLIDisplayInfo("Quaternion source set to: " + value);
        ws.send(JSON.stringify({ action: "setQuaternionSource", source: value }));
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
