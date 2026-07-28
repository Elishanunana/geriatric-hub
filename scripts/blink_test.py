from gpiozero import LED
from time import sleep

led = LED(17)
print("Blinking LED on GPIO 17. Press Ctrl+C to stop.")

try:
    while True:
        led.on()
        print("ON")
        sleep(0.5)
        led.off()
        print("OFF")
        sleep(0.5)
except KeyboardInterrupt:
    led.off()
    print("\nDone.")
