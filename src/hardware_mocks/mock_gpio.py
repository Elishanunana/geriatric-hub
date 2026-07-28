"""
mock_gpio.py
============
Terminal-based simulator for the Raspberry Pi 4 GPIO peripherals
(Sections 3.4.4 and 3.4.5 of the project report):

  • Single-channel relay        — appliance switching (light / fan)
  • Normally-open push button   — physical SOS trigger
  • Bi-colour LED               — system status indicator

Behavior Contract
-----------------
The real GPIOController (control_logic/gpio_controller.py) will use
RPi.GPIO with these public methods:
  - .set_relay(state: bool)
  - .on_button_press(callback)            — register SOS-button handler
  - .set_led(color, mode)                 — color: 'green'|'red'; mode: 'off'|'steady'|'flashing'
  - .start()                              — begin button polling
  - .stop()                               — clean shutdown / GPIO.cleanup()

This mock preserves all of those signatures, plus adds:
  - .trigger_sos()                        — programmatic button-press injection
  - .simulate_button_interactive()        — interactive button injection

Replacement Plan
----------------
Swap the import:
    from hardware_mocks.mock_gpio import MockGPIOController as GPIOController
to the real RPi.GPIO-backed driver. No changes in the SOS handler.

Author: Wise (Asumang Pobi Godwin) - KNUST COE 497
"""

import threading
import logging
from datetime import datetime, timezone
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — these mirror the LED status semantics from Section 3.4.5
# ---------------------------------------------------------------------------
LED_OFF       = "off"
LED_STEADY    = "steady"
LED_FLASHING  = "flashing"

LED_GREEN     = "green"
LED_RED       = "red"


# ---------------------------------------------------------------------------
# MockGPIOController
# ---------------------------------------------------------------------------

class MockGPIOController:
    """
    Print-driven simulation of the relay, SOS button, and bi-colour LED.
    """

    def __init__(self):
        self._relay_state: bool = False                 # False = OFF, True = ON
        self._led_color: str = LED_GREEN
        self._led_mode: str = LED_STEADY                # default: steady green = normal operation
        self._button_callback: Optional[Callable[[], None]] = None
        self._lock = threading.Lock()
        self._started = False

        # Print initial banner so the user knows the LED's default state.
        self._print_led_state()

    # -- Lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """
        Real implementation: configure GPIO pin modes, attach interrupt for
        the SOS button, and begin LED PWM if used. Mock: no-op marker.
        """
        with self._lock:
            self._started = True
        logger.info("MockGPIOController started.")

    def stop(self) -> None:
        """Real implementation: GPIO.cleanup(). Mock: print and reset."""
        with self._lock:
            self._started = False
        self.set_relay(False)
        self.set_led(LED_GREEN, LED_OFF)
        print("[GPIO]  ⏹   Controller stopped, all pins released.")

    # -- Relay --------------------------------------------------------------

    def set_relay(self, state: bool) -> None:
        """
        Drive the relay to ON (True) or OFF (False).
        Real implementation drives a GPIO output pin through a transistor
        which controls the relay coil (Section 3.4.4).
        """
        with self._lock:
            previous = self._relay_state
            self._relay_state = bool(state)

        if previous == self._relay_state:
            return  # No-op — already in requested state.

        symbol = "💡 ON " if self._relay_state else "💤 OFF"
        print(f"[GPIO]  {symbol}  Relay → {'CLOSED (appliance powered)' if self._relay_state else 'OPEN (appliance off)'}")

    @property
    def relay_state(self) -> bool:
        """Current relay state (True = ON)."""
        return self._relay_state

    # -- LED ----------------------------------------------------------------

    def set_led(self, color: str, mode: str) -> None:
        """
        Configure the bi-colour status LED.

        Per Section 3.4.5:
          • steady green     → normal operation
          • flashing green   → active listening
          • steady red       → system fault
          • flashing red     → SOS sent, awaiting caregiver acknowledgement
          • off              → no power / startup
        """
        if color not in (LED_GREEN, LED_RED):
            logger.warning("set_led: invalid color %r", color)
            return
        if mode not in (LED_OFF, LED_STEADY, LED_FLASHING):
            logger.warning("set_led: invalid mode %r", mode)
            return

        with self._lock:
            if self._led_color == color and self._led_mode == mode:
                return
            self._led_color = color
            self._led_mode = mode

        self._print_led_state()

    def _print_led_state(self) -> None:
        if self._led_mode == LED_OFF:
            symbol = "⚫"
            description = "OFF"
        elif self._led_color == LED_GREEN and self._led_mode == LED_STEADY:
            symbol = "🟢"
            description = "STEADY GREEN — normal operation"
        elif self._led_color == LED_GREEN and self._led_mode == LED_FLASHING:
            symbol = "🟢⚡"
            description = "FLASHING GREEN — active listening"
        elif self._led_color == LED_RED and self._led_mode == LED_STEADY:
            symbol = "🔴"
            description = "STEADY RED — system fault"
        elif self._led_color == LED_RED and self._led_mode == LED_FLASHING:
            symbol = "🔴⚡"
            description = "FLASHING RED — SOS sent, awaiting acknowledgement"
        else:
            symbol = "❓"
            description = f"{self._led_color}/{self._led_mode}"

        print(f"[GPIO]  {symbol}  LED → {description}")

    # -- Push button (SOS trigger) ------------------------------------------

    def on_button_press(self, callback: Callable[[], None]) -> None:
        """
        Register a callback to be invoked when the physical SOS button is
        pressed. In production, this is wired to a GPIO falling-edge
        interrupt (Section 3.4.5).
        """
        self._button_callback = callback
        logger.info("SOS button callback registered.")

    def trigger_sos(self) -> None:
        """
        Programmatic SOS-button press injection. Mirrors a falling edge on
        the physical pin. Used by integration tests and by the dev console.
        """
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds") + "Z"
        print(f"\n[GPIO]  🆘  SOS BUTTON PRESSED  @ {ts}")

        if self._button_callback is None:
            logger.warning("trigger_sos: no callback registered — event dropped.")
            print("[GPIO]      (no handler registered — event dropped)")
            return

        try:
            self._button_callback()
        except Exception:
            logger.exception("SOS button callback raised")

    def simulate_button_interactive(self) -> None:
        """Interactive variant: prompts the user to press Enter to fire."""
        try:
            input("[GPIO]  Press ENTER to simulate SOS button press (Ctrl+C to cancel) ")
        except (EOFError, KeyboardInterrupt):
            print("[GPIO]  ✗  Cancelled.")
            return
        self.trigger_sos()


# ---------------------------------------------------------------------------
# Standalone test harness
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    gpio = MockGPIOController()
    gpio.start()

    # Relay test
    gpio.set_relay(True)
    gpio.set_relay(False)

    # LED state walk-through
    gpio.set_led(LED_GREEN, LED_FLASHING)   # listening
    gpio.set_led(LED_RED, LED_FLASHING)     # SOS pending ack
    gpio.set_led(LED_GREEN, LED_STEADY)     # back to normal

    # SOS button test
    def _on_sos():
        print("   ↳ SOS handler invoked — would dispatch SMS + flash red LED")
    gpio.on_button_press(_on_sos)
    gpio.trigger_sos()

    gpio.stop()
