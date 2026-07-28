from gpiozero import LED, Button
from signal import pause

led = LED(17)
button = Button(27, pull_up=True)

button.when_pressed = led.on
button.when_released = led.off

print("Press and hold the button to light the LED. Ctrl+C to stop.")
pause()
