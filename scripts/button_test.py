from gpiozero import Button
from time import sleep

button = Button(27, pull_up=True)
print("Press the button. Ctrl+C to stop.")

try:
    while True:
        if button.is_pressed:
            print("PRESSED")
            sleep(0.2)
        else:
            pass
        sleep(0.05)
except KeyboardInterrupt:
    print("\nDone.")
