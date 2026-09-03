import json
import asyncio
from bleak import BleakClient, BleakScanner
from langchain_core.tools import tool

# Configuration
UNO_DEVICE_NAME = "Dhruv_Uno_R4"
UNO_MAC_ADDRESS = None
SERVICE_UUID = "19B10000-E8F2-537E-4F6C-D104768A1214"
CHAR_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214"

class BLEManager:
    def __init__(self):
        self.client = None
        # NEW: A hardware execution lock to queue simultaneous tool calls
        self.hardware_queue = asyncio.Lock() 

    async def connect(self):
        if self.client and self.client.is_connected:
            return True
            
        print("\n[BLE] Scanning for Dhruv's Arduino Uno R4...")
        device = await BleakScanner.find_device_by_name(UNO_DEVICE_NAME)

        if not device:
            print("[BLE Error] Arduino not found.")
            return False

        print(f"[BLE] Found {device.name}. Establishing persistent connection...")
        self.client = BleakClient(device)
        await self.client.connect()
        print("[BLE] Connected successfully! Actuators ready.")
        return True

    async def send_payload_and_wait(self, payload: dict, duration_ms: int) -> str:
        """Queues the command, transmits it, and locks execution for the duration."""
        
        # If LangGraph fires 3 tools at once, they wait here in line automatically
        async with self.hardware_queue:
            if not await self.connect():
                return "Failed: Arduino BLE is disconnected or out of range."
            
            try:
                json_str = json.dumps(payload) + "\n"
                data_bytes = json_str.encode('utf-8')
                
                await self.client.write_gatt_char(CHAR_UUID, data_bytes, response=True)
                print(f"[BLE TX] -> {json_str.strip()}")
                
                # Wait for the physical action to finish before releasing the lock for the next command!
                await asyncio.sleep(duration_ms / 1000.0)
                
                return f"Successfully executed: {json_str.strip()}"
                
            except Exception as e:
                print(f"[BLE Error] Transmission failed: {e}")
                return f"Hardware transmission error: {str(e)}"

# Singleton instance
ble_manager = BLEManager()

@tool
async def spin_vehicle(degrees: int, direction: str) -> str:
    """
    Spins the rover in place by a specific degree amount.
    Args:
        degrees: The amount to spin (e.g., 90, 180, 360).
        direction: 'L' (Left/Counter-Clockwise) or 'R' (Right/Clockwise).
    """
    clean_direction = direction.upper()[:1]
    
    # Calculate physical duration estimate to hold the Python thread lock
    # (Assuming calibration: 360 degrees takes ~3000 ms at full speed)
    estimated_duration_ms = int((degrees / 360.0) * 3000)
    
    payload = {
        "c": "SPN",
        "d": clean_direction,
        "deg": degrees,
        "t": estimated_duration_ms
    }
    
    return await ble_manager.send_payload_and_wait(payload, estimated_duration_ms)

@tool
async def diagonal_movement(direction: str, duration_ms: int, speed: int = 255) -> str:
    """
    Moves the rover diagonally (requires Mecanum/Omni wheels).
    Args:
        direction: 'FL' (Forward-Left), 'FR' (Forward-Right), 'BL' (Backward-Left), 'BR' (Backward-Right).
        duration_ms: Milliseconds to move. Max 5000ms.
        speed: Motor speed 0-255.
    """
    safe_duration = min(duration_ms, 5000)
    safe_speed = max(0, min(speed, 255))
    clean_direction = direction.upper()[:2]
    
    payload = {
        "c": "DIA",
        "d": clean_direction,
        "t": safe_duration,
        "s": safe_speed
    }
    
    return await ble_manager.send_payload_and_wait(payload, safe_duration)

@tool
async def drive_vehicle(direction: str, duration_ms: int, speed: int = 255) -> str:
    """
    Moves the entire rover in a specific direction.
    Args:
        direction: 'F' (Forward), 'B' (Backward), 'L' (Left), 'R' (Right).
        duration_ms: Milliseconds to move. Max limit is 5000ms.
        speed: PWM Motor speed from 0 to 255.
    """
    safe_duration = min(duration_ms, 5000) 
    safe_speed = max(0, min(speed, 255))
    clean_direction = direction.upper()[:1] 
    
    payload = {
        "c": "DRV",
        "d": clean_direction,
        "t": safe_duration,
        "s": safe_speed
    }
    
    # Pass the duration so the manager knows exactly how long to lock the queue
    return await ble_manager.send_payload_and_wait(payload, safe_duration)

@tool
async def control_motor(motor_id: int, direction: str, duration_ms: int, speed: int = 255) -> str:
    """
    Controls a single specific motor individually.
    Args:
        motor_id: 1, 2, 3, or 4.
        direction: 'F' (Forward) or 'B' (Backward).
        duration_ms: Milliseconds to run the motor.
        speed: PWM Motor speed from 0 to 255.
    """
    safe_duration = min(duration_ms, 5000)
    safe_speed = max(0, min(speed, 255))
    clean_direction = direction.upper()[:1]
    
    payload = {
        "c": "MOT",
        "id": motor_id,
        "d": clean_direction,
        "t": safe_duration,
        "s": safe_speed
    }
    
    return await ble_manager.send_payload_and_wait(payload, safe_duration)

@tool
async def emergency_stop() -> str:
    """Immediately kills all PWM signals to halt the rover."""
    payload = {"c": "STP"}
    return await ble_manager.send_payload_and_wait(payload, 0)