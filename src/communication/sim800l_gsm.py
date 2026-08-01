"""
sim800l_gsm.py
==============
Real SIM800L GSM driver: a hardware-backed replacement for
hardware_mocks.MockGSMModule that talks to the physical module over UART
using the EXACT AT-command sequence proven end-to-end in
scripts/sms_test.py (real SMS delivered to a Ghanaian number, CSQ 14,
+CMGS: 55 OK).

It preserves MockGSMModule's public contract byte-for-byte, so the swap in
main.py is a single import-line alias change and nothing downstream
(SOSHandler, sms_payload_handler) needs editing:

    # from src.hardware_mocks.mock_sim800l import MockGSMModule
    from src.communication.sim800l_gsm import SIM800LGSMModule as MockGSMModule

Public contract preserved from MockGSMModule
--------------------------------------------
    .send_sms(recipient, body)            -> bool
    .read_unread_messages()               -> List[InboundSMS]
    .delete_message(index)                -> bool
    .inject_inbound_sms(sender, body)     -> InboundSMS
    .simulate_inbound_interactive()       -> Optional[InboundSMS]
    .signal_strength()                    -> int   (CSQ scale)
    .storage_used()                       -> int
    .sent_log  (property)                 -> List[OutboundSMS]

The InboundSMS / OutboundSMS dataclasses are imported from the mock so the
records this driver produces are the SAME types the rest of the system and
the integration tests already consume.

Validation status (honest scoping for the defense)
---------------------------------------------------
* OUTBOUND (send_sms, signal_strength): the AT sequence here is the one
  validated end-to-end on real hardware via scripts/sms_test.py. This is
  the safety-critical SOS path and it is proven.
* INBOUND (read_unread_messages, delete_message): implemented against the
  standard text-mode AT+CMGL / AT+CMGD commands, but NOT yet bench-tested on
  the physical module. The Wi-Fi REST channel is the primary caregiver->hub
  sync path; SMS inbound is the fallback. Bench-test these before relying on
  them, and keep the mock available for that layer if time is short.

Two deliberate design decisions
-------------------------------
1. Lazy serial + locked UART. The port is opened on first use, not at
   construction, so the hub boots even if the module is unplugged and this
   file imports on a laptop with no pyserial. Every AT transaction is guarded
   by a lock because a physical SOS (button thread) and a voice SOS (listener
   thread) can fire concurrently, and two overlapping writes to one UART would
   corrupt each other.
2. SOS attempts transmission even when network registration is marginal.
   scripts/sms_test.py aborts if AT+CREG? is not registered; for a
   safety-critical alert that is the wrong behaviour. This driver checks
   registration, waits briefly if needed, then attempts the send regardless
   and lets the module's +CMGS / ERROR response decide the outcome.

Wiring (unchanged from the proven script): SIM800L on /dev/serial0 at 9600
baud, TX/RX crossed, powered from the LM2596 buck converter at ~4.0 V.

Author: Wise (Akabua Elisha Nunana) - KNUST COE 497
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, List, Optional

# Reuse the SAME record dataclasses the mock defines, so events produced by
# this real driver are contract-identical to the mock's downstream.
from src.hardware_mocks.mock_sim800l import InboundSMS, OutboundSMS

logger = logging.getLogger(__name__)

# Defaults proven in scripts/sms_test.py
DEFAULT_PORT = "/dev/serial0"
DEFAULT_BAUD = 9600
CTRL_Z = bytes([26])   # terminates the SMS body for AT+CMGS


def _utc_ts() -> str:
    """UTC timestamp in the exact format MockGSMModule uses."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds") + "Z"


class SIM800LGSMModule:
    """
    Hardware SIM800L driver, drop-in for MockGSMModule. See module docstring
    for the full contract and validation scoping.
    """

    def __init__(
        self,
        sim_storage_capacity: int = 30,
        *,
        port: str = DEFAULT_PORT,
        baud: int = DEFAULT_BAUD,
        boot_delay: float = 2.0,
        registration_wait: float = 10.0,
        send_timeout: float = 30.0,
        retry_cooldown: float = 8.0,
    ):
        """
        Parameters
        ----------
        sim_storage_capacity : int
            Kept for interface parity with the mock (caps the local injected
            queue used by tests). The real SIM's own storage is managed by
            delete_message after processing.
        port : str
            UART device. Default /dev/serial0 (as validated).
        baud : int
            Baud rate. Default 9600 (as validated).
        boot_delay : float
            Seconds to wait after opening the port before the first AT, giving
            the module time to settle. Matches sms_test.py's 2 s.
        registration_wait : float
            Seconds to wait for network registration before attempting a send
            when AT+CREG? is not yet registered.
        send_timeout : float
            Seconds to wait for the +CMGS confirmation after Ctrl-Z.
        retry_cooldown : float
            Minimum seconds between connection attempts after a failed one, so
            the 30s background pollers don't spam the log while the module is
            unavailable. A successful connection clears this.
        """
        self._port = port
        self._baud = baud
        self._boot_delay = boot_delay
        self._registration_wait = registration_wait
        self._send_timeout = send_timeout
        self._capacity = sim_storage_capacity

        self._serial: Any = None            # pyserial handle, opened lazily
        # Only pyserial being absent is permanent (it can't appear at runtime).
        # A failed open or AT handshake is treated as TRANSIENT — the module
        # may still be powering up or registering — so we allow retries rather
        # than latching GSM off for the whole session after one hiccup.
        self._pyserial_missing = False
        self._last_connect_attempt = 0.0
        self._retry_cooldown = retry_cooldown

        # One lock for ALL UART access — concurrent SOS triggers must not
        # interleave AT transactions on the shared serial line. Connection
        # setup runs under this same lock (see _ensure_serial), so background
        # pollers and the SOS path can never collide during open/handshake.
        self._serial_lock = threading.RLock()

        self._sent_log: List[OutboundSMS] = []
        self._log_lock = threading.Lock()

        # Local injected-inbound queue, so inject_inbound_sms /
        # simulate_inbound_interactive keep working (tests, dev console) even
        # against the real class. Injected indices live in a high range to
        # avoid colliding with real SIM slot indices.
        self._injected: Deque[InboundSMS] = deque()
        self._next_injected_index = 900001

    # -- Serial lifecycle ---------------------------------------------------

    def _ensure_serial(self) -> bool:
        """
        Ensure the UART is open and the module is in SMS text mode. Returns
        False (and logs) on failure, without raising, so the hub stays up.

        Thread-safety: the ENTIRE open + AT + CMGF handshake runs under
        _serial_lock with double-checked locking. Multiple background threads
        (SyncEngine, SMSPayloadHandler) and the SOS path can call this
        concurrently; only the first performs the handshake, the rest reuse
        the handle. Without this, two threads could open/mutate self._serial
        at once and one would read from a half-closed port.
        """
        # Fast path: already connected (no lock needed for a plain reference read).
        if self._serial is not None:
            return True
        # Permanent-only short-circuit: pyserial can't appear mid-run.
        if self._pyserial_missing:
            return False

        with self._serial_lock:
            # Re-check now that we hold the lock — another thread may have
            # connected (or failed) while we were waiting.
            if self._serial is not None:
                return True
            if self._pyserial_missing:
                return False

            # Cooldown: don't re-attempt (and re-spam logs) too soon after a
            # failure. The 30s pollers would otherwise retry every cycle.
            now = time.monotonic()
            if (now - self._last_connect_attempt) < self._retry_cooldown \
                    and self._last_connect_attempt > 0.0:
                return False
            self._last_connect_attempt = now

            try:
                import serial  # lazy: keeps this module importable w/o pyserial
            except ImportError:
                self._pyserial_missing = True   # permanent
                logger.error("pyserial not installed — run: python -m pip install pyserial")
                print("[GSM]  X  pyserial not installed (pip install pyserial)")
                return False

            try:
                ser = serial.Serial(self._port, self._baud, timeout=1)
            except Exception as exc:  # noqa: BLE001 - transient; allow retry
                logger.error("Failed to open %s: %s", self._port, exc)
                print(f"[GSM]  X  Failed to open {self._port}: {exc}")
                print("[GSM]     Is serial enabled in raspi-config? Is wiring crossed?")
                return False

            time.sleep(self._boot_delay)
            self._serial = ser

            # Confirm the module answers, then set text mode. Still under the
            # lock, so self._serial cannot be nulled by another thread here.
            if "OK" not in self._send_at("AT"):
                logger.error("SIM800L not responding to AT — check TX/RX wiring.")
                print("[GSM]  X  Module not responding to AT (check crossed TX/RX).")
                try:
                    ser.close()
                except Exception:  # noqa: BLE001
                    pass
                self._serial = None           # transient — retry after cooldown
                return False

            self._send_at("AT+CMGF=1")        # SMS text mode
            logger.info("SIM800L ready on %s @ %d baud.", self._port, self._baud)
            print(f"[GSM]  OK  SIM800L ready on {self._port} @ {self._baud} baud.")
            return True

    def close(self) -> None:
        """Release the serial port. Optional; the mock has no equivalent, so
        main.py does not call this, but it is here for clean shutdown/tests."""
        with self._serial_lock:
            if self._serial is not None:
                try:
                    self._serial.close()
                except Exception:  # noqa: BLE001
                    pass
                self._serial = None

    # -- Low-level AT helper (mirrors sms_test.py's send_at) -----------------

    def _send_at(self, command: str, wait_seconds: float = 1.5, quiet: bool = True) -> str:
        """Write one AT command and return the decoded response. Assumes the
        caller already holds _serial_lock (or that serial is single-threaded
        at open time)."""
        if self._serial is None:
            return ""
        if not quiet:
            print(f"[GSM]  >> {command}")
        self._serial.write((command + "\r\n").encode())
        time.sleep(wait_seconds)
        response = self._serial.read_all().decode(errors="ignore").strip()
        if not quiet:
            for line in response.splitlines():
                if line.strip():
                    print(f"[GSM]  << {line}")
        return response

    # -- Outbound API -------------------------------------------------------

    def send_sms(self, recipient: str, body: str) -> bool:
        """
        Send one SMS via AT+CMGS using the validated sequence. Returns True on
        a +CMGS/OK confirmation, False otherwise. Records the attempt (success
        or failure) in sent_log, matching MockGSMModule.
        """
        if not recipient or not body:
            logger.warning("send_sms: empty recipient or body — rejected.")
            return False

        if not self._ensure_serial():
            self._record_outbound(recipient, body, success=False)
            print(f"[GSM]  X  Cannot send to {recipient}: serial unavailable.")
            return False

        with self._serial_lock:
            # 1. Signal + registration check. For SOS we attempt the send even
            #    if registration is marginal (see module docstring), but we
            #    still surface the state.
            csq = self._read_csq_locked()
            reg = self._send_at("AT+CREG?")
            registered = (",1" in reg) or (",5" in reg)
            if not registered and self._registration_wait > 0:
                print(f"[GSM]  …  Not yet registered (CSQ={csq}); "
                      f"waiting {self._registration_wait:g}s...")
                time.sleep(self._registration_wait)
                reg = self._send_at("AT+CREG?")
                registered = (",1" in reg) or (",5" in reg)
            if not registered:
                logger.warning("SIM800L not registered; attempting send anyway (SOS).")
                print("[GSM]  !  Still not registered — attempting send regardless.")

            # 2. Ensure text mode, then the CMGS handshake.
            self._send_at("AT+CMGF=1")
            self._send_at(f'AT+CMGS="{recipient}"', wait_seconds=2.0)

            # 3. Body + Ctrl-Z terminator.
            self._serial.write(body.encode())
            time.sleep(0.5)
            self._serial.write(CTRL_Z)

            # 4. Wait for +CMGS ... OK (or ERROR), up to send_timeout.
            success = self._await_send_confirmation_locked()

        self._record_outbound(recipient, body, success=success)
        self._print_outbound_banner(recipient, body, success)
        return success

    def _await_send_confirmation_locked(self) -> bool:
        """Poll the serial line for the CMGS result. Caller holds the lock."""
        deadline = time.time() + self._send_timeout
        buffer = ""
        while time.time() < deadline:
            chunk = self._serial.read_all().decode(errors="ignore")
            if chunk:
                buffer += chunk
                if "+CMGS:" in buffer and "OK" in buffer:
                    return True
                if "ERROR" in buffer:
                    logger.error("SIM800L returned ERROR on send.")
                    return False
            time.sleep(0.5)
        logger.warning("send_sms: timed out waiting for +CMGS confirmation.")
        return False

    def _record_outbound(self, recipient: str, body: str, success: bool) -> None:
        rec = OutboundSMS(recipient=recipient, body=body,
                          sent_at=_utc_ts(), success=success)
        with self._log_lock:
            self._sent_log.append(rec)

    def _print_outbound_banner(self, recipient: str, body: str, success: bool) -> None:
        """Same visual style as the mock, so demo output is consistent."""
        border = "=" * 70
        status = "SENT" if success else "FAILED"
        print(f"\n[GSM] {border}")
        print(f"[GSM]  OUTBOUND SMS  ->  {recipient}   [{status}]")
        print(f"[GSM]      Time: {_utc_ts()}")
        print(f"[GSM]      Body ({len(body)} chars):")
        for line in (body.splitlines() or [body]):
            print(f"[GSM]        {line}")
        print(f"[GSM] {border}\n")

    @property
    def sent_log(self) -> List[OutboundSMS]:
        """All outbound dispatch attempts — for test assertions."""
        with self._log_lock:
            return list(self._sent_log)

    # -- Diagnostics --------------------------------------------------------

    def signal_strength(self) -> int:
        """Return the CSQ RSSI value (0-31; 99 = unknown). -1 if unavailable."""
        if not self._ensure_serial():
            return -1
        with self._serial_lock:
            return self._read_csq_locked()

    def _read_csq_locked(self) -> int:
        """Parse AT+CSQ -> '+CSQ: <rssi>,<ber>'. Caller holds the lock."""
        resp = self._send_at("AT+CSQ")
        try:
            marker = resp.index("+CSQ:") + len("+CSQ:")
            rssi = resp[marker:].strip().split(",")[0].strip()
            return int(rssi)
        except (ValueError, IndexError):
            return -1

    # -- Inbound API (implemented; not yet hardware-validated) --------------

    def read_unread_messages(self) -> List[InboundSMS]:
        """
        Return unread inbound messages: any locally injected ones plus, if the
        module is reachable, messages parsed from AT+CMGL="REC UNREAD".

        NOTE: the AT+CMGL parsing path has not been bench-tested on the
        physical module. Validate before relying on it (see module docstring).
        """
        messages: List[InboundSMS] = list(self._injected)
        if not self._ensure_serial():
            return messages
        with self._serial_lock:
            resp = self._send_at('AT+CMGL="REC UNREAD"', wait_seconds=2.0)
        messages.extend(self._parse_cmgl(resp))
        return messages

    @staticmethod
    def _parse_cmgl(resp: str) -> List[InboundSMS]:
        """
        Parse a text-mode AT+CMGL response. Each message is a header line
            +CMGL: <index>,"REC UNREAD","<sender>",,"<timestamp>"
        followed by one body line.
        """
        out: List[InboundSMS] = []
        lines = resp.splitlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith("+CMGL:"):
                try:
                    header = line[len("+CMGL:"):].strip()
                    parts = header.split(",")
                    index = int(parts[0].strip())
                    sender = parts[2].strip().strip('"') if len(parts) > 2 else ""
                    ts = parts[4].strip().strip('"') if len(parts) > 4 else _utc_ts()
                    body = lines[i + 1].strip() if i + 1 < len(lines) else ""
                    out.append(InboundSMS(index=index, sender=sender,
                                          body=body, received_at=ts))
                    i += 2
                    continue
                except (ValueError, IndexError):
                    logger.warning("Could not parse CMGL header: %r", line)
            i += 1
        return out

    def delete_message(self, index: int) -> bool:
        """
        Delete a message by index. Injected messages are removed locally;
        real SIM-slot messages are removed with AT+CMGD=<index>.
        """
        for msg in list(self._injected):
            if msg.index == index:
                self._injected.remove(msg)
                print(f"[GSM]  Deleted injected message at index {index}")
                return True
        if not self._ensure_serial():
            return False
        with self._serial_lock:
            resp = self._send_at(f"AT+CMGD={index}")
        ok = "OK" in resp
        if ok:
            print(f"[GSM]  Deleted SIM message at index {index}")
        else:
            logger.warning("delete_message: AT+CMGD=%d did not return OK", index)
        return ok

    def inject_inbound_sms(self, sender: str, body: str) -> InboundSMS:
        """Programmatic inbound injection for tests / dev console. Does not
        touch hardware; the injected message is returned by
        read_unread_messages until deleted."""
        if len(self._injected) >= self._capacity:
            dropped = self._injected.popleft()
            logger.warning("Injected queue full — dropped index %d", dropped.index)
        msg = InboundSMS(index=self._next_injected_index, sender=sender,
                         body=body, received_at=_utc_ts())
        self._next_injected_index += 1
        self._injected.append(msg)
        print(f"\n[GSM]  INBOUND (injected)  <-  {sender}  (index={msg.index})")
        print(f"[GSM]      Body: {body}\n")
        return msg

    def simulate_inbound_interactive(self) -> Optional[InboundSMS]:
        """Interactive inbound injection via terminal prompts."""
        try:
            print("\n[GSM]  Simulate inbound SMS")
            sender = input("[GSM]    Sender phone number: ").strip()
            print("[GSM]    Enter body (single line, then ENTER):")
            body = input("[GSM]    > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("[GSM]  Cancelled.")
            return None
        if not sender or not body:
            print("[GSM]  Empty sender or body — aborted.")
            return None
        return self.inject_inbound_sms(sender, body)

    def storage_used(self) -> int:
        """Number of locally injected unread messages (parity with the mock)."""
        return len(self._injected)


# ---------------------------------------------------------------------------
# Standalone test harness — send one real SMS, mirroring sms_test.py.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Send one real SMS via SIM800L.")
    parser.add_argument("recipient", nargs="?", default="+233200510903",
                        help="Destination phone number (E.164, e.g. +233...).")
    parser.add_argument("--body", default="SOS test from the geriatric care hub "
                        "(SIM800LGSMModule). If you got this, the driver works.")
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    args = parser.parse_args()

    gsm = SIM800LGSMModule(port=args.port, baud=args.baud)
    print(f"Signal strength (CSQ): {gsm.signal_strength()}")
    ok = gsm.send_sms(args.recipient, args.body)
    print(f"\nResult: {'SUCCESS' if ok else 'FAILED'}")
    gsm.close()
    raise SystemExit(0 if ok else 1)
