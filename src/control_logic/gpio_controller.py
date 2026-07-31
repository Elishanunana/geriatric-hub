"""
gpio_controller.py
==================
Real Raspberry Pi GPIO driver, a drop-in replacement for
hardware_mocks.MockGPIOController. It preserves the mock's public contract
byte-for-byte, so the swap in main.py is a single import-line alias change
and the SOS handler / command dispatcher need no edits:

    # from src.hardware_mocks.mock_gpio import MockGPIOController
    from src.control_logic.gpio_controller import GPIOController as MockGPIOController

Backed by gpiozero, using the exact pin setup validated on the bench:
  * scripts/blink_test.py      -> LED on GPIO 17 (.on()/.off())
  * scripts/button_test.py     -> Button on GPIO 27, pull_up=True
  * scripts/button_led_test.py -> event-driven Button.when_pressed pattern

Hardware mapping (one LED wired, so it is assigned deliberately)
----------------------------------------------------------------
  * GPIO 17 LED  == the APPLIANCE output. set_relay(True) lights it,
    set_relay(False) turns it off. This is the "Sɔ kanea no / Dum kanea no"
    demo action. (The report calls this a relay; the physical relay module
    is deferred over the 3.3 V / 5 V logic issue, so the LED is the
    defensible actuator on the same control pathway.)
  * GPIO 27 button == the physical SOS trigger. A falling edge fires the
    registered SOS callback, so a button press dispatches a real SMS exactly
    like a voice "Boa me!".
  * The bi-colour STATUS LED (set_led: green/red, steady/flashing) is NOT
    physically wired, so set_led keeps the mock's console output. Wire a
    second LED later and this is where you would drive it.

Public contract preserved from MockGPIOController
-------------------------------------------------
    .set_relay(state: bool)          -> None
    .relay_state  (property)         -> bool
    .set_led(color, mode)            -> None
    .on_button_press(callback)       -> None
    .trigger_sos()                   -> None   (programmatic / dev console)
    .simulate_button_interactive()   -> None
    .start()                         -> None
    .stop()                          -> None
plus the LED_* / colour constants, re-exported from the mock so any caller
importing them from either module gets the same values.

gpiozero is imported LAZILY inside start()/_ensure_hw(), so this file imports
fine on a laptop with no GPIO stack. If gpiozero or the pins are unavailable,
the driver DEGRADES to console output (like the mock) instead of crashing --
the dev-console SOS injection and the 30/30 integration tests keep working.

Author: Wise (Akabua Elisha Nunana) - KNUST COE 497
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Optional

# Re-use the mock's LED/colour constants so both modules agree on values.
from src.hardware_mocks.mock_gpio import (
    LED_OFF,
    LED_STEADY,
    LED_FLASHING,
    LED_GREEN,
    LED_RED,
)

logger = logging.getLogger(__name__)

# Validated pin assignments (BCM numbering).
APPLIANCE_LED_PIN = 17     # blink_test.py
SOS_BUTTON_PIN = 27        # button_test.py
BUTTON_BOUNCE_S = 0.15     # debounce; SOSHandler also applies its own cooldown


class GPIOController:
    """
    Real GPIO driver (drop-in for MockGPIOController). See module docstring
    for the pin mapping and the graceful-degradation behaviour.
    """

    def __init__(self):
        self._relay_state: bool = False
        self._led_color: str = LED_GREEN
        self._led_mode: str = LED_STEADY
        self._button_callback: Optional[Callable[[], None]] = None
        self._lock = threading.Lock()
        self._started = False

        # gpiozero handles, created lazily. _hw_ok is tri-state: None = not yet
        # attempted, True = real GPIO active, False = degraded to console.
        self._led: Any = None
        self._button: Any = None
        self._hw_ok: Optional[bool] = None

        # Match the mock: announce the default LED state on construction.
        self._print_led_state()

    # -- Hardware bring-up --------------------------------------------------

    def _ensure_hw(self) -> bool:
        """Lazily create the LED output. Returns True if real GPIO is usable,
        False if we must degrade to console output. Never raises."""
        if self._hw_ok is not None:
            return self._hw_ok
        try:
            from gpiozero import LED
        except Exception as exc:  # noqa: BLE001 - ImportError or pin-factory issue
            logger.warning("gpiozero unavailable (%s) — GPIO runs in console mode.", exc)
            print("[GPIO]  !  gpiozero unavailable — running in console (mock-like) mode.")
            self._hw_ok = False
            return False
        try:
            self._led = LED(APPLIANCE_LED_PIN)
            self._led.off()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not init LED on GPIO %d (%s) — console mode.",
                           APPLIANCE_LED_PIN, exc)
            print(f"[GPIO]  !  Could not init LED on GPIO {APPLIANCE_LED_PIN} — console mode.")
            self._hw_ok = False
            return False
        self._hw_ok = True
        return True

    # -- Lifecycle ----------------------------------------------------------

    def start(self) -> None:
        """Bring up GPIO: ensure the LED, then attach the SOS button's
        falling-edge handler. Degrades to console mode if hardware is absent."""
        with self._lock:
            self._started = True

        if not self._ensure_hw():
            logger.info("GPIOController started (console mode — no physical GPIO).")
            return

        try:
            from gpiozero import Button
            self._button = Button(SOS_BUTTON_PIN, pull_up=True,
                                  bounce_time=BUTTON_BOUNCE_S)
            self._button.when_pressed = self._on_hardware_button
            logger.info("GPIOController started — LED on GPIO %d, SOS button on GPIO %d.",
                        APPLIANCE_LED_PIN, SOS_BUTTON_PIN)
            print(f"[GPIO]  OK  Real GPIO active — appliance LED=GPIO{APPLIANCE_LED_PIN}, "
                  f"SOS button=GPIO{SOS_BUTTON_PIN}.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not init SOS button on GPIO %d (%s).",
                           SOS_BUTTON_PIN, exc)
            print(f"[GPIO]  !  Could not init SOS button on GPIO {SOS_BUTTON_PIN} — "
                  f"voice SOS still works.")

    def stop(self) -> None:
        """Turn everything off and release the pins (gpiozero cleanup)."""
        with self._lock:
            self._started = False
        self.set_relay(False)
        self.set_led(LED_GREEN, LED_OFF)
        try:
            if self._button is not None:
                self._button.close()
            if self._led is not None:
                self._led.off()
                self._led.close()
        except Exception:  # noqa: BLE001
            pass
        self._button = None
        self._led = None
        print("[GPIO]  Controller stopped, all pins released.")

    # -- Relay (== appliance LED on GPIO 17) --------------------------------

    def set_relay(self, state: bool) -> None:
        """Drive the appliance LED ON (True) / OFF (False)."""
        with self._lock:
            previous = self._relay_state
            self._relay_state = bool(state)
        if previous == self._relay_state:
            return  # no-op; already in requested state

        if self._ensure_hw() and self._led is not None:
            try:
                self._led.on() if self._relay_state else self._led.off()
            except Exception as exc:  # noqa: BLE001
                logger.warning("LED write failed (%s).", exc)

        symbol = "ON " if self._relay_state else "OFF"
        state_txt = "CLOSED (appliance powered)" if self._relay_state else "OPEN (appliance off)"
        print(f"[GPIO]  {symbol}  Appliance LED (GPIO{APPLIANCE_LED_PIN}) -> {state_txt}")

    @property
    def relay_state(self) -> bool:
        return self._relay_state

    # -- Status LED (console-only; no bi-colour LED wired) ------------------

    def set_led(self, color: str, mode: str) -> None:
        """Set the system status indicator. No physical bi-colour LED is
        wired, so this reflects state to the console exactly as the mock does.
        Wiring a second LED? Drive it here."""
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
            symbol, desc = "( )", "OFF"
        elif self._led_color == LED_GREEN and self._led_mode == LED_STEADY:
            symbol, desc = "[G]", "STEADY GREEN — normal operation"
        elif self._led_color == LED_GREEN and self._led_mode == LED_FLASHING:
            symbol, desc = "[G*]", "FLASHING GREEN — active listening"
        elif self._led_color == LED_RED and self._led_mode == LED_STEADY:
            symbol, desc = "[R]", "STEADY RED — system fault"
        elif self._led_color == LED_RED and self._led_mode == LED_FLASHING:
            symbol, desc = "[R*]", "FLASHING RED — SOS sent, awaiting acknowledgement"
        else:
            symbol, desc = "[?]", f"{self._led_color}/{self._led_mode}"
        print(f"[GPIO]  {symbol}  status LED -> {desc}")

    # -- SOS push button ----------------------------------------------------

    def on_button_press(self, callback: Callable[[], None]) -> None:
        """Register the handler fired on a physical SOS-button press."""
        self._button_callback = callback
        logger.info("SOS button callback registered.")

    def _on_hardware_button(self) -> None:
        """gpiozero falling-edge handler (runs on gpiozero's own thread)."""
        self._dispatch_sos(source="hardware button")

    def trigger_sos(self) -> None:
        """Programmatic SOS-button press (dev console / integration tests)."""
        self._dispatch_sos(source="injected")

    def _dispatch_sos(self, source: str) -> None:
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds") + "Z"
        print(f"\n[GPIO]  SOS BUTTON PRESSED ({source})  @ {ts}")
        if self._button_callback is None:
            logger.warning("SOS press but no callback registered — event dropped.")
            print("[GPIO]      (no handler registered — event dropped)")
            return
        try:
            self._button_callback()
        except Exception:  # noqa: BLE001 - isolate downstream faults
            logger.exception("SOS button callback raised")

    def simulate_button_interactive(self) -> None:
        """Interactive variant: press Enter to fire an SOS."""
        try:
            input("[GPIO]  Press ENTER to simulate SOS button press (Ctrl+C to cancel) ")
        except (EOFError, KeyboardInterrupt):
            print("[GPIO]  Cancelled.")
            return
        self.trigger_sos()


# ---------------------------------------------------------------------------
# Standalone test harness — run on the Pi to exercise real GPIO in isolation.
#   python -m src.control_logic.gpio_controller
# LED should turn on for 2s; then press the physical button to fire the SOS
# callback. Ctrl+C to exit.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    gpio = GPIOController()
    gpio.start()

    print("\n--- Appliance LED test (GPIO 17) ---")
    gpio.set_relay(True)
    time.sleep(2)
    gpio.set_relay(False)

    print("\n--- LED status walk-through (console) ---")
    gpio.set_led(LED_GREEN, LED_FLASHING)
    gpio.set_led(LED_RED, LED_FLASHING)
    gpio.set_led(LED_GREEN, LED_STEADY)

    print("\n--- SOS button test (GPIO 27) ---")

    def _on_sos():
        print("   -> SOS handler invoked — would dispatch SMS + flash red LED")

    gpio.on_button_press(_on_sos)
    print("Press the physical SOS button now (Ctrl+C to finish)...")
    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        gpio.stop()
        