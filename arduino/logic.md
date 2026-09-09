# Dhruv: Hardware Communication Logic

This document explains the architecture of how the Mac Server (The Master/Brain) commands the Arduino Uno R4 (The Slave/Body) over Bluetooth Low Energy (BLE).

## 1. The Master-Slave Architecture
- **Master (Mac Server):** Runs the heavy AI models, parses the camera feed from the Jetson Nano, and executes the "Agentic Body Loop" (Loop B). 
- **Slave (Arduino Uno):** Simply a dumb motor controller. It knows nothing about AI or cameras. Its only job is to listen for a JSON string over BLE, spin the motors for a requested amount of time, and then wait for the next command.

## 2. Is it Streaming?
**Yes and No.**
The Mac runs a continuous background task that "pulses" commands to the Arduino. 
For example, if Dhruv decides to wander forward, the Mac will send a Drive command: `{"c": "DRV", "d": "F", "t": 500, "s": 150}`. 

Because the time `t` is only `500` milliseconds, the Mac will wait 500ms and then immediately send the exact same command again, over and over, effectively "streaming" continuous movement to the Arduino as long as the state remains `WANDERING`.

### Why use time-bound commands (`t`)?
This is a safety feature. If the Mac simply told the Arduino to "Start driving forward indefinitely", and then the Mac crashed or the Bluetooth disconnected, the robot would crash into a wall forever. 
Instead, the Mac says "Drive forward for 500 milliseconds". The Arduino spins the wheels for 500ms and then automatically stops. If the Mac is healthy, it will send the next 500ms command right as the first one finishes, creating a smooth, continuous drive.

## 3. The Commands & Expected Behavior

When the Arduino receives a JSON string over BLE, it must parse the `"c"` (Command) key to understand what to do.

### A. Drive Vehicle (`"c": "DRV"`)
**Payload:** `{"c": "DRV", "d": "F", "t": 500, "s": 150}`
- **`d` (Direction):** `"F"` (Forward), `"B"` (Backward), `"L"` (Strafe Left), `"R"` (Strafe Right).
- **`t` (Time):** Duration in milliseconds (e.g., 500). Max 5000ms.
- **`s` (Speed):** PWM Speed from 0-255.
**Behavior:** The Arduino should write the directional pins HIGH for all 4 wheels (if 4WD), apply the PWM speed, `delay(t)`, and then write all pins LOW.

### B. Spin Vehicle (`"c": "SPN"`)
**Payload:** `{"c": "SPN", "d": "L", "deg": 90, "t": 750}`
- **`d` (Direction):** `"L"` (Counter-clockwise), `"R"` (Clockwise).
- **`deg` (Degrees):** The physical degrees the AI wants the robot to turn.
- **`t` (Time):** The Mac estimates how long this spin will take based on calibration.
**Behavior:** The Arduino should spin the left wheels backward and the right wheels forward (or vice versa), `delay(t)`, and then write all pins LOW. (The Arduino can ignore `deg` and just rely on `t`).

### C. Emergency Stop (`"c": "STP"`)
**Payload:** `{"c": "STP"}`
**Behavior:** The Arduino should instantly write LOW to all motor pins to halt the robot immediately. There is no delay here.

### D. Diagonal Movement (`"c": "DIA"`)
**Payload:** `{"c": "DIA", "d": "FL", "t": 1000, "s": 200}`
- **`d` (Direction):** `"FL"` (Forward-Left), `"FR"`, `"BL"`, `"BR"`.
**Behavior:** Only applicable if using Mecanum or Omni wheels. The Arduino calculates which specific wheels to spin to achieve diagonal strafing, `delay(t)`, and then stops.

### E. Individual Motor Control (`"c": "MOT"`)
**Payload:** `{"c": "MOT", "id": 1, "d": "F", "t": 200, "s": 255}`
- **`id` (Motor ID):** 1, 2, 3, or 4.
**Behavior:** The Arduino spins only that specific motor for `t` milliseconds. Used for precise micro-adjustments or debugging.

## 4. Writing the C++ Code
To implement this logic on the Arduino, you will need to:
1. Include `<ArduinoJson.h>`.
2. Inside the BLE receive loop, pass the string to `deserializeJson()`.
3. Use a series of `if/else if` statements checking `doc["c"]`.
4. Trigger the appropriate pins connected to your L298N (or similar) motor driver!
