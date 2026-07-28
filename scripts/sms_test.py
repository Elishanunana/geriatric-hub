"""
sms_test.py — Send one SMS through the SIM800L.

This is the "hello world" of GSM. If this works, the entire SOS
pathway of the project is proven functional.
"""

import serial
import time

CAREGIVER_PHONE = "+233200510903"
SERIAL_DEVICE = "/dev/serial0"
BAUD_RATE = 9600

def send_at(ser, command, wait_seconds=1.5, quiet=False):
    """Send an AT command and return the response."""
    if not quiet:
        print(f"  >> {command}")
    ser.write((command + "\r\n").encode())
    time.sleep(wait_seconds)
    response = ser.read_all().decode(errors="ignore").strip()
    if not quiet:
        for line in response.splitlines():
            if line.strip():
                print(f"  << {line}")
    return response

def main():
    print("=" * 60)
    print("  SIM800L SMS TEST")
    print("=" * 60)

    print(f"\nOpening serial port {SERIAL_DEVICE} at {BAUD_RATE} baud...")
    try:
        ser = serial.Serial(SERIAL_DEVICE, BAUD_RATE, timeout=1)
        time.sleep(2)
    except Exception as e:
        print(f"Failed to open serial: {e}")
        print("Did you enable serial in raspi-config?")
        return 1

    print("\n--- Step 1: Check module is responding ---")
    response = send_at(ser, "AT")
    if "OK" not in response:
        print("\nModule not responding. Check TX/RX wiring (they should be crossed).")
        ser.close()
        return 1

    print("\n--- Step 2: Check signal strength ---")
    response = send_at(ser, "AT+CSQ")

    print("\n--- Step 3: Check network registration ---")
    response = send_at(ser, "AT+CREG?")
    if ",1" not in response and ",5" not in response:
        print("\nNot registered on network yet. Waiting 10 seconds and retrying...")
        time.sleep(10)
        response = send_at(ser, "AT+CREG?")
        if ",1" not in response and ",5" not in response:
            print("Still not registered. Check antenna, SIM card, signal.")
            ser.close()
            return 1

    print("\n--- Step 4: Set SMS text mode ---")
    send_at(ser, "AT+CMGF=1")

    print("\n--- Step 5: Send the SMS ---")
    print(f"Sending to {CAREGIVER_PHONE}...")
    send_at(ser, f'AT+CMGS="{CAREGIVER_PHONE}"', wait_seconds=2)
    message = "SOS test from the geriatric care hub. If you received this, the project works."
    ser.write(message.encode())
    time.sleep(0.5)
    ser.write(bytes([26]))
    print("Message sent. Waiting up to 30 seconds for confirmation...")

    deadline = time.time() + 30
    while time.time() < deadline:
        response = ser.read_all().decode(errors="ignore")
        if response.strip():
            for line in response.splitlines():
                if line.strip():
                    print(f"  << {line}")
            if "+CMGS:" in response and "OK" in response:
                print("\nSMS sent successfully. Check your phone.")
                ser.close()
                return 0
            if "ERROR" in response:
                print("\nSMS failed. Check credit balance and phone number format.")
                ser.close()
                return 1
        time.sleep(0.5)

    print("\nTimed out waiting for confirmation. SMS may still arrive.")
    ser.close()
    return 0

if __name__ == "__main__":
    exit(main())
