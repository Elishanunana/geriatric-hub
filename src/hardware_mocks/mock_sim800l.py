"""
mock_sim800l.py
===============
Terminal-based simulator for the SIM800L GSM module (Sections 3.4.3, 3.5.2,
and 3.5.4 of the project report).

The real SIM800L is bidirectional:
  • Outbound — emergency SOS SMS dispatch (Section 3.5.2 SOS pathway).
  • Inbound  — caregiver-app payloads polled at 30s intervals via AT
               commands (Section 3.5.2 SMS payload handler).

Behavior Contract
-----------------
The real GSMModule class will expose:
  - .send_sms(recipient, body)           → bool
  - .read_unread_messages()              → List[InboundSMS]
  - .delete_message(index)               → None
  - .signal_strength()                   → int (CSQ scale)

This mock preserves all of those signatures, plus adds:
  - .inject_inbound_sms(sender, body)    — programmatic injection (tests)
  - .simulate_inbound_interactive()      — interactive terminal injection

Replacement Plan
----------------
Swap the import:
    from hardware_mocks.mock_sim800l import MockGSMModule as GSMModule
to the real driver. The SMS handler in control_logic/sms_payload_handler.py
needs no changes.

Author: Wise (Asumang Pobi Godwin) - KNUST COE 497
"""

import threading
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional, Deque

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class InboundSMS:
    """One unread message retrieved from the SIM800L's storage."""
    index: int            # SIM message slot index — used by .delete_message()
    sender: str           # Originating phone number
    body: str             # Message body (UTF-8 decoded)
    received_at: str      # ISO-8601 UTC timestamp of arrival


@dataclass
class OutboundSMS:
    """Record of an outbound SMS dispatch — kept for test inspection."""
    recipient: str
    body: str
    sent_at: str
    success: bool


# ---------------------------------------------------------------------------
# MockGSMModule
# ---------------------------------------------------------------------------

class MockGSMModule:
    """
    Simulates the SIM800L via terminal printing (outbound) and a programmatic
    or interactive inbound queue.
    """

    def __init__(self, sim_storage_capacity: int = 30):
        """
        Parameters
        ----------
        sim_storage_capacity : int
            Mirrors the real module's limited message memory. The handler
            in control_logic/sms_payload_handler.py is responsible for calling
            .delete_message() after processing to prevent overflow.
        """
        self._inbound: Deque[InboundSMS] = deque()
        self._sent_log: List[OutboundSMS] = []
        self._next_index = 1
        self._capacity = sim_storage_capacity
        self._lock = threading.Lock()

    # -- Outbound API -------------------------------------------------------

    def send_sms(self, recipient: str, body: str) -> bool:
        """
        Dispatch an outbound SMS. In production this issues
        AT+CMGS to the SIM800L over UART at 9600 baud (Section 3.4.3).

        Returns True on success — the mock always succeeds unless the body
        is empty.
        """
        if not recipient or not body:
            logger.warning("send_sms: empty recipient or body — rejected.")
            return False

        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds") + "Z"
        record = OutboundSMS(recipient=recipient, body=body, sent_at=ts, success=True)

        with self._lock:
            self._sent_log.append(record)

        # Visible terminal output — outbound SMS is a high-importance event
        # (especially for SOS), so it gets a clear, multi-line banner.
        border = "═" * 70
        print(f"\n[GSM] {border}")
        print(f"[GSM]  📡  OUTBOUND SMS  →  {recipient}")
        print(f"[GSM]      Time: {ts}")
        print(f"[GSM]      Body ({len(body)} chars):")
        for line in body.splitlines() or [body]:
            print(f"[GSM]        {line}")
        print(f"[GSM] {border}\n")
        return True

    @property
    def sent_log(self) -> List[OutboundSMS]:
        """All outbound dispatches — for test assertions."""
        return list(self._sent_log)

    # -- Inbound API --------------------------------------------------------

    def read_unread_messages(self) -> List[InboundSMS]:
        """
        Returns all unread inbound messages. Mirrors the real handler's
        AT+CMGL="REC UNREAD" call. Messages remain in storage until
        explicitly deleted via .delete_message().
        """
        with self._lock:
            return list(self._inbound)

    def delete_message(self, index: int) -> bool:
        """
        Remove a specific message by SIM-storage index. Mirrors AT+CMGD.
        Called by the SMS payload handler after successful processing
        (Section 3.5.2: "Cleanup — the processed SMS is deleted...").
        """
        with self._lock:
            for msg in list(self._inbound):
                if msg.index == index:
                    self._inbound.remove(msg)
                    print(f"[GSM]  🗑   Deleted message at index {index}")
                    return True
        logger.warning("delete_message: index %d not found", index)
        return False

    def inject_inbound_sms(self, sender: str, body: str) -> InboundSMS:
        """
        Programmatic inbound SMS injection. Used by integration tests to
        deliver a payload to the SMS handler deterministically.
        """
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds") + "Z"
        with self._lock:
            if len(self._inbound) >= self._capacity:
                # Mirror the real module's behavior: oldest dropped.
                dropped = self._inbound.popleft()
                logger.warning("SIM storage full — dropped index %d", dropped.index)

            msg = InboundSMS(
                index=self._next_index,
                sender=sender,
                body=body,
                received_at=ts,
            )
            self._next_index += 1
            self._inbound.append(msg)

        print(f"\n[GSM]  📥  INBOUND SMS RECEIVED  ←  {sender}  (index={msg.index})")
        print(f"[GSM]      Body: {body}\n")
        return msg

    def simulate_inbound_interactive(self) -> Optional[InboundSMS]:
        """
        Interactive inbound SMS injection via terminal prompts.
        Use during manual testing of the SMS payload handler pipeline.
        """
        try:
            print("\n[GSM]  ▼ Simulate inbound SMS")
            sender = input("[GSM]    Sender phone number: ").strip()
            print("[GSM]    Enter body (single line, then ENTER):")
            body = input("[GSM]    > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("[GSM]  ✗  Cancelled.")
            return None

        if not sender or not body:
            print("[GSM]  ✗  Empty sender or body — aborted.")
            return None

        return self.inject_inbound_sms(sender, body)

    # -- Diagnostics --------------------------------------------------------

    def signal_strength(self) -> int:
        """
        Mock CSQ value (0–31, RSSI scale). Real handler may use this to
        decide whether to defer transmission.
        """
        return 22   # "good signal" placeholder

    def storage_used(self) -> int:
        with self._lock:
            return len(self._inbound)


# ---------------------------------------------------------------------------
# Standalone test harness
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    gsm = MockGSMModule()

    # Outbound test (e.g., SOS dispatch)
    gsm.send_sms("+233244123456", "SOS: Help requested by elder. 14:32 UTC.")

    # Inbound test (programmatic + interactive)
    gsm.inject_inbound_sms("+233244555555", "MED|INSERT|Paracetamol|500mg|08:00|HMAC=abc123")

    print("\nNow simulating an interactive inbound message...")
    gsm.simulate_inbound_interactive()

    print(f"\nUnread queue length: {gsm.storage_used()}")
    for m in gsm.read_unread_messages():
        print(f"  • [{m.index}] from {m.sender}: {m.body}")