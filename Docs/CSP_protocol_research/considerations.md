# Project README: Advanced Communication Protocol for Raspberry Pi and Arduino

## 1. Introduction

The primary goal of this project is to reliably transmit high-frequency orientation data (quaternions) from the Raspberry Pi to the Arduino for real-time control applications.

This system moves beyond simple serial data streaming via UART protocol by considering the implementation of a structured communication protocol inspired by the **Cubesat Space Protocol (CSP)**.

## 2. Established Foundation: UART Communication

A stable physical communication link has already been established and verified between the two devices. The key characteristics of this foundation are:

*   **Physical Layer:** UART (Universal Asynchronous Receiver-Transmitter).
*   **Hardware:** The Raspberry Pi's primary hardware UART (`/dev/serial0`) is connected to the Arduino Nano's hardware UART (Pins 0 and 1).
*   **Logic Level Safety:** A bidirectional logic level shifter is used to safely interface the Raspberry Pi's 3.3V logic with the Arduino's 5V logic.
*   **Performance:** The Raspberry Pi has been configured to use its high-performance PL011 UART on the GPIO pins (`dtoverlay=miniuart-bt`).

With this reliable byte-stream connection established, the next step is to implement a protocol that gives this data structure scalability, meaning and reliability.

## 3. The Proposed Protocol: Cubesat Space Protocol (CSP)

### 3.1. General Information about CSP

The Cubesat Space Protocol (CSP) is an open-source, lightweight network stack designed for embedded systems, particularly in the aerospace industry for satellite communication. It is not merely a packet format but a complete networking solution that provides features typically found in larger systems like TCP/IP.

Key features of CSP include:
*   **Network-Layer Routing:** CSP allows packets to be routed between different nodes in a network.
*   **Addressing:** Each device on the network has a unique address, and applications on each device can be addressed via "ports."
*   **Reliable Transport:** It includes an optional reliable transport protocol (RDP) that handles acknowledgements and retransmissions, ensuring guaranteed delivery.
*   **Interface Independence:** It is designed to run over various physical layers, including UART, I2C, CAN bus, and radio links.

### 3.2. How CSP is Usually Implemented in Industry

A powerful "Edge Computer" (like the Raspberry Pi) would run the full `libcsp` stack. This gateway would then communicate with simpler "Endpoint Devices" (like the Arduino). If the endpoint device is too resource-constrained to run the full CSP stack, the gateway acts as a **protocol translator**. It receives full CSP packets from its main application and translates them into a simpler, more lightweight protocol (like Modbus or a custom binary format) to send over the final link to the endpoint. This design provides the power of a full network while respecting the limitations of the simpler hardware.

## 4. Constraints and Proposed Solution for Our Project

While using the full, official `libcsp` library is feasible on the Raspberry Pi, it presents a significant challenge for our Arduino Nano.

*   **The Constraint:** The Arduino Nano, with only 2 KB of SRAM and 32 KB of Flash memory, does not have the resources to run the full `libcsp` library. The memory requirements for its buffers, routing tables, and code size are too large.

*   **The Proposed Solution:** We will adopt the professional Gateway pattern by creating a **CSP-inspired simplified protocol**. This approach gives us the best of both worlds:
    1.  **On the Raspberry Pi:** We can choose to either use the full `libcsp` library for our main application and write a separate "gateway" program for translation, or simply use a powerful Python script that constructs our simplified packets.
    2.  **On the Arduino:** We will implement a lightweight, memory-efficient protocol that borrows the key concepts from CSP.

*   **The Problems:** The Arduino in our case is simulating a central main computer between the other subsystems. It should be a node in our system, it should be the main CSP runner. By implementing a lightweight protocol on it it crosses the logic behind it being the main computer. At last it makes the Rasberry our main computer, but the raspberry is merely a proccesing unit for a star tracker.