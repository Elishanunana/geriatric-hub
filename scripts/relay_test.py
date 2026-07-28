from gpiozero import OutputDevice
from time import sleep

relay = OutputDevice(22, active_high=True, initial_value=False)
print("Testing relay on GPIO 22. Ctrl+C to stop.")

try:
    while True:
        print("Relay ON (you should hear a click)")
        relay.on()
        sleep(2)
        print("Relay OFF (another click)")
        relay.off()
        sleep(2)
except KeyboardInterrupt:
    relay.off()
    print("\nDone.")
