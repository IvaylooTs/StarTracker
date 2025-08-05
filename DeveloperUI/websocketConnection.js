function TerminalDisplayInfo(message){
    const messageDiv = document.getElementById('messages');
    const p = document.createElement('p');
    p.textContent = message;
    messageDiv.appendChild(p);
    messageDiv.scrollTop = messageDiv.scrollHeight;
}

const quaternionOutput = document.getElementById('quaternionOutput');
const systemStats = document.getElementById('systemStats');

let ws = null;
let connectionRetryInterval = null;

let firstLogin = false

function ConnectToWebSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        return; // Already connected or trying
    }
    try {
        ws = new WebSocket('ws://192.168.55.160:6789');
    } catch (e) {
        console.error("Failed to create WebSocket:", e);
        TerminalDisplayInfo("WebSocket creation failed.");
        return; // Prevent running rest of connect logic
    }
    // ws = new WebSocket('ws://192.168.55.160:6789');

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
                systemStats.textContent = `CPU Usage: ${msg.CPU_temp}°C\nMemory Usage: ${msg.CPU_usage}%\nRam usage: ${msg.RAM_percent}%`;

                console.log({x, y, z, w});
                window.Satellite3DObject.quaternion.set(x, y, z, w);
                AddDataToChart({ w, x, y, z });
            }
        } catch (e) {
            console.warn('Invalid or partial JSON:', e);
        }
    };

    ws.onclose = () => {
        TerminalDisplayInfo("Connection lost. Retrying...");
        console.warn("WebSocket closed.");
        retryWebSocketConnection();
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

function sendCommand(cmd) {
    if (ws.readyState === WebSocket.OPEN) {
        if (!confirm("Are you sure you want to send command: \"" + cmd + "\"?"))
            return;

        if(cmd === 'calibrate') {
            ws.send(JSON.stringify({ action: cmd }));
        } else if(cmd === 'addOffset') {
            const w = parseFloat(document.getElementById("q_w").value);
            const x = parseFloat(document.getElementById("q_x").value);
            const y = parseFloat(document.getElementById("q_y").value);
            const z = parseFloat(document.getElementById("q_z").value);

            if ([w, x, y, z].some(isNaN)) {
                alert("Please enter all quaternion values (w, x, y, z).");
                return;
            }
            ws.send(JSON.stringify({ action: cmd, data: {w, x, y, z} }));
        } else if(cmd === "lostInSpace") {
            ws.send(JSON.stringify({ action: cmd }));
        }
    } else {
        alert("WebSocket not connected");
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
