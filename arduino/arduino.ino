#include <ArduinoBLE.h>

BLEService dhruvService("19B10000-E8F2-537E-4F6C-D104768A1214"); // Dhruv BLE Service

// BLE Characteristic - custom 128-bit UUID, read and writable by central
// Using a large string buffer (256 bytes) to hold the JSON payloads sent by the Mac
BLEStringCharacteristic commandChar("19B10001-E8F2-537E-4F6C-D104768A1214", BLERead | BLEWrite, 256);

void setup() {
  Serial.begin(115200);
  
  // Wait for Serial to initialize (you can comment this out if running on battery)
  while (!Serial);

  // Begin BLE initialization
  if (!BLE.begin()) {
    Serial.println("Starting BLE failed!");
    while (1);
  }

  // Set the advertised local name. This MUST match 'UNO_DEVICE_NAME' in tools/actuators.py!
  BLE.setLocalName("Dhruv_Uno_R4");
  BLE.setAdvertisedService(dhruvService);

  // Add the characteristic to the service
  dhruvService.addCharacteristic(commandChar);

  // Add the service
  BLE.addService(dhruvService);

  // Start advertising the BLE signal
  BLE.advertise();

  Serial.println("Dhruv_Uno_R4 BLE Active.");
  Serial.println("Waiting for connections from Mac Server...");
}

void loop() {
  // Listen for the Mac Server to connect
  BLEDevice central = BLE.central();

  // If the Mac connects:
  if (central) {
    Serial.print("Connected to Mac Server: ");
    Serial.println(central.address());

    // While the Mac remains connected:
    while (central.connected()) {
      
      // Check if the Mac wrote a new JSON command to the characteristic
      if (commandChar.written()) {
        // Read the string value
        String commandJSON = commandChar.value();
        
        Serial.print("Received BLE Command: ");
        Serial.println(commandJSON);
        
        // ---------------------------------------------------------
        // TODO: Include ArduinoJson library here to parse commandJSON
        // and trigger your L298N/Motor Shield pins!
        // ---------------------------------------------------------
      }
    }

    // When the Mac disconnects:
    Serial.print(F("Disconnected from Mac Server: "));
    Serial.println(central.address());
  }
}
