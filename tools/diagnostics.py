import os
import httpx
from tools.actuators import ble_manager

async def verify_arduino_ble() -> bool:
    """Verifies BLE connectivity to Arduino Uno R4 on startup."""
    print("  [1/3] Scanning for Arduino Uno R4 BLE ('Dhruv_Uno_R4')...")
    connected = await ble_manager.connect()
    if connected:
        print("        --> Arduino Uno R4: ONLINE (BLE Connected & Ready)")
        return True
    else:
        print("        --> Arduino Uno R4: OFFLINE (Check power, antenna, or device name)")
        return False

async def verify_moondream_endpoint() -> bool:
    """Verifies that the Moondream API Key is configured."""
    api_key = os.getenv("MOONDREAM_API_KEY")
    api_url = os.getenv("MOONDREAM_API_URL")
    
    if not api_key:
        print("  [2/3] Moondream API Key: MISSING in .env")
        return False
    if not api_url:
        print("  [2/3] Moondream API URL: MISSING in .env")
        return False

    print("  [2/3] Moondream Cloud API: CONFIGURED (Ready)")
    return True

def verify_openai_key() -> bool:
    """Verifies that the primary LLM API key exists."""
    key = os.getenv("AZURE_OPENAI_API_KEY")
    if key and key.startswith("Cz0"):
        print("  [3/3] OpenAI API Key: CONFIGURED")
        return True
    print("  [3/3] OpenAI API Key: MISSING OR INVALID")
    return False

async def run_system_diagnostics():
    """Runs full hardware & service pre-flight checklist."""
    print("\n" + "=" * 60)
    print(" Dhruv Core System Diagnostics (Hardware Pre-Flight Check)")
    print("=" * 60)
    
    ble_ok = await verify_arduino_ble()
    vision_ok = await verify_moondream_endpoint()
    llm_ok = verify_openai_key()
    
    print("=" * 60)
    if ble_ok and vision_ok and llm_ok:
        print(" SYSTEM HEALTH: OPTIMAL - All master-slave links operational.\n")
    else:
        print(" SYSTEM HEALTH: DEGRADED - Review warnings above before inferencing.\n")