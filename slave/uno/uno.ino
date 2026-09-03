#include <ArduinoBLE.h>
#include <ArduinoJson.h>

// BLE Configuration matching the Python actuator tools
const char* BLE_DEVICE_NAME = "Dhruv_Uno_R4";
const char* SERVICE_UUID    = "19B10000-E8F2-537E-4F6C-D104768A1214";
const char* CHAR_UUID       = "19B10001-E8F2-537E-4F6C-D104768A1214";

BLEService roverService(SERVICE_UUID);
// 128 byte buffer is plenty for our compact JSON payloads
BLECharacteristic commandChar(CHAR_UUID, BLERead | BLEWrite, 128); 

// Motor Pin Definitions (Adjust these to match your wiring)
// Note: Uno R4 PWM pins are ~3, 5, 6, 9, 10, 11
struct Motor {
  int pinIN1;
  int pinIN2;
  int pinEN;  // PWM pin for speed control
  unsigned long stopTime; // Tracks when this motor should automatically halt
};

// Map your 4 motors (e.g., Front-Left, Rear-Left, Front-Right, Rear-Right)
Motor motors[4] = {
  {2, 4, 3, 0},  // Motor 1: IN1=2, IN2=4, EN=3 (PWM)
  {7, 8, 5, 0},  // Motor 2: IN1=7, IN2=8, EN=5 (PWM)
  {12, 13, 6, 0},// Motor 3: IN1=12, IN2=13, EN=6 (PWM)
  {A0, A1, 9, 0} // Motor 4: IN1=A0, IN2=A1, EN=9 (PWM)
};

void setup() {
  Serial.begin(115200);
  
  // Initialize all motor pins to OUTPUT and halt them
  for (int i = 0; i < 4; i++) {
    pinMode(motors[i].pinIN1, OUTPUT);
    pinMode(motors[i].pinIN2, OUTPUT);
    pinMode(motors[i].pinEN, OUTPUT);
    haltMotor(i);
  }

  // Initialize BLE
  if (!BLE.begin()) {
    Serial.println("Starting BLE failed!");
    while (1);
  }

  BLE.setLocalName(BLE_DEVICE_NAME);
  BLE.setAdvertisedService(roverService);
  roverService.addCharacteristic(commandChar);
  BLE.addService(roverService);
  BLE.advertise();

  Serial.println("Dhruv BLE Actuator Node Active. Waiting for connections...");
}

void loop() {
  BLEDevice central = BLE.central();

  if (central) {
    Serial.print("Connected to master brain: ");
    Serial.println(central.address());

    while (central.connected()) {
      // 1. Check for incoming BLE commands
      if (commandChar.written()) {
        processCommand(commandChar.value(), commandChar.valueLength());
      }

      // 2. Continuous non-blocking check to halt expired motor commands
      unsigned long currentMillis = millis();
      for (int i = 0; i < 4; i++) {
        if (motors[i].stopTime > 0 && currentMillis >= motors[i].stopTime) {
          haltMotor(i);
          motors[i].stopTime = 0; // Reset timer
        }
      }
    }
    
    // Safety fallback: if Bluetooth disconnects, kill all motors immediately
    Serial.println("Master brain disconnected. Emergency halt.");
    executeEmergencyStop();
  }
}

// --- Logic Processors ---

void processCommand(const uint8_t* payload, int length) {
  // Convert byte array to a standard String
  String jsonStr = "";
  for (int i = 0; i < length; i++) {
    jsonStr += (char)payload[i];
  }
  
  Serial.print("Received: ");
  Serial.println(jsonStr);

  // Parse JSON
  JsonDocument doc;
  DeserializationError error = deserializeJson(doc, jsonStr);

  if (error) {
    Serial.print("JSON Parse failed: ");
    Serial.println(error.c_str());
    return;
  }

  const char* cmdType = doc["c"];

  if (strcmp(cmdType, "STP") == 0) {
    executeEmergencyStop();
  } 
  else if (strcmp(cmdType, "MOT") == 0) {
    int id = doc["id"];      // 1 to 4
    const char* dir = doc["d"]; // "F" or "B"
    int duration = doc["t"];
    int speed = doc["s"];
    
    if (id >= 1 && id <= 4) {
      runMotor(id - 1, dir[0], speed, duration);
    }
  }
  else if (strcmp(cmdType, "DRV") == 0) {
    const char* dir = doc["d"];
    int duration = doc["t"];
    int speed = doc["s"];
    
    executeDifferentialDrive(dir[0], speed, duration);
  }
}

// --- Motor Hardware Controllers ---

void executeEmergencyStop() {
  for (int i = 0; i < 4; i++) {
    haltMotor(i);
    motors[i].stopTime = 0;
  }
  Serial.println("EMERGENCY STOP EXECUTED");
}

void executeDifferentialDrive(char dir, int speed, int duration) {
  // Assuming differential drive mapping:
  // Left side: Motor 0 & 1 | Right side: Motor 2 & 3
  
  if (dir == 'F') {
    // All forward
    runMotor(0, 'F', speed, duration); runMotor(1, 'F', speed, duration);
    runMotor(2, 'F', speed, duration); runMotor(3, 'F', speed, duration);
  } 
  else if (dir == 'B') {
    // All backward
    runMotor(0, 'B', speed, duration); runMotor(1, 'B', speed, duration);
    runMotor(2, 'B', speed, duration); runMotor(3, 'B', speed, duration);
  }
  else if (dir == 'L') {
    // Pivot Left (Left backward, Right forward)
    runMotor(0, 'B', speed, duration); runMotor(1, 'B', speed, duration);
    runMotor(2, 'F', speed, duration); runMotor(3, 'F', speed, duration);
  }
  else if (dir == 'R') {
    // Pivot Right (Left forward, Right backward)
    runMotor(0, 'F', speed, duration); runMotor(1, 'F', speed, duration);
    runMotor(2, 'B', speed, duration); runMotor(3, 'B', speed, duration);
  }
}

void runMotor(int index, char dir, int speed, int duration) {
  // Set direction
  if (dir == 'F') {
    digitalWrite(motors[index].pinIN1, HIGH);
    digitalWrite(motors[index].pinIN2, LOW);
  } else if (dir == 'B') {
    digitalWrite(motors[index].pinIN1, LOW);
    digitalWrite(motors[index].pinIN2, HIGH);
  }

  // Set speed
  analogWrite(motors[index].pinEN, speed);

  // Set non-blocking timeout
  motors[index].stopTime = millis() + duration;
}

void haltMotor(int index) {
  digitalWrite(motors[index].pinIN1, LOW);
  digitalWrite(motors[index].pinIN2, LOW);
  analogWrite(motors[index].pinEN, 0);
}