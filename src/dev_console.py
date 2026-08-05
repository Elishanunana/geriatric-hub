"""
dev_console.py
==============
Interactive REPL for driving a running hub from a separate terminal.
Communicates with the hub via the DevCommandQueue table — the hub's
DevCommandPoller drains the queue and applies each command against its
live in-memory mocks.

Usage
-----
Terminal A:
    python -m src.main

Terminal B:
    python -m src.dev_console

Then type commands in Terminal B and watch them take effect in Terminal A.
The console is intended for developers and for the academic defense demo
— it is NOT user-facing functionality.

Author: Wise (Asumang Pobi Godwin) — KNUST COE 497
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.data_management.db_connection import get_connection


# ===========================================================================
# Constants
# ===========================================================================

BANNER_RULE = "═" * 70
WAIT_FOR_RESULT_TIMEOUT = 5.0    # seconds to wait for the hub to apply a command


# ===========================================================================
# Low-level command queue I/O
# ===========================================================================

def enqueue_command(command: str, args: Optional[Dict[str, Any]] = None) -> int:
    """Insert a command into DevCommandQueue. Returns the new cmd_id."""
    args_json = json.dumps(args or {}, ensure_ascii=False)
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO DevCommandQueue (command, args_json) VALUES (?, ?);",
            (command, args_json),
        )
        return cur.lastrowid


def wait_for_result(cmd_id: int, timeout: float = WAIT_FOR_RESULT_TIMEOUT) -> Optional[str]:
    """
    Poll DevCommandQueue until the hub marks `cmd_id` as applied. Returns
    the result string or None if the timeout expires (which usually means
    the hub isn't running).
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT applied_at, result FROM DevCommandQueue WHERE cmd_id = ?;",
                (cmd_id,),
            ).fetchone()
        if row and row["applied_at"]:
            return row["result"]
        time.sleep(0.2)
    return None


def send(command: str, args: Optional[Dict[str, Any]] = None) -> None:
    """Enqueue, wait, print outcome."""
    cmd_id = enqueue_command(command, args)
    result = wait_for_result(cmd_id)
    if result is None:
        print(f"  ⚠  No response from hub within {WAIT_FOR_RESULT_TIMEOUT}s.")
        print(f"     Is `python -m src.main` running in another terminal?")
        return
    if result.startswith("error:"):
        print(f"  ✗  {result}")
    else:
        print(f"  ✓  {result}")


# ===========================================================================
# Connectivity check
# ===========================================================================

def hub_is_running() -> bool:
    """
    Send a PING command and wait briefly for a response. The DevCommandPoller
    only runs when the hub is up, so a successful pong proves the hub is alive.
    """
    cmd_id = enqueue_command("PING")
    return wait_for_result(cmd_id, timeout=2.0) == "pong"


# ===========================================================================
# Interactive menu
# ===========================================================================

MENU = """
  [1]  Toggle Wi-Fi caregiver connection (CONNECT)
  [2]  Toggle Wi-Fi caregiver connection (DISCONNECT)
  [3]  Inject voice command  →  Boa me!  (SOS)
  [4]  Inject voice command  →  Yε, mafa m'aduru  (DOSE_CONFIRMED)
  [5]  Inject voice command  →  Mfaa m'aduru nkaa  (DOSE_MISSED)
  [6]  Inject voice command  →  Sua fitaa no  (APPLIANCE_ON)
  [7]  Inject voice command  →  Sua fitaa no na  (APPLIANCE_OFF)
  [8]  Inject voice command  →  Aduru bεn na mefa?  (READ_SCHEDULE)
  [9]  Inject voice command  →  Mesrε wo, yε san bio  (REPEAT_LAST)
  [10] Trigger physical SOS button (GPIO)
  [11] Inject inbound SMS — caregiver schedule update (mock-HMAC)
  [12] Inject inbound SMS — UNKNOWN sender (should be rejected)
  [13] Inject inbound SMS — MALFORMED payload (should be rejected)
  [14] Custom voice action  (free-text)
  [15] Custom inbound SMS   (free-text)

  [q]  Quit
"""


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(
        timespec="milliseconds"
    ).replace("+00:00", "Z")


def _build_mock_sms_payload(
    *,
    drug_name: str,
    dosage: str,
    time_due: str,
    elder_id: int = 1,
) -> str:
    """Build a MOCK-signed inbound SMS payload (matches Task 7 format)."""
    return (
        f"MED|INSERT|{uuid.uuid4()}|{elder_id}|{drug_name}|{dosage}|"
        f"{time_due}|DAILY|1|{_utcnow_iso()}|HMAC=MOCK"
    )


def handle_choice(choice: str) -> bool:
    """Apply one menu choice. Returns False to quit, True to continue."""
    choice = choice.strip().lower()

    if choice in ("q", "quit", "exit"):
        return False

    if choice == "1":
        send("TOGGLE_WIFI", {"connected": True})

    elif choice == "2":
        send("TOGGLE_WIFI", {"connected": False})

    elif choice == "3":
        send("INJECT_VOICE", {"action": "SOS"})

    elif choice == "4":
        send("INJECT_VOICE", {"action": "DOSE_CONFIRMED"})

    elif choice == "5":
        send("INJECT_VOICE", {"action": "DOSE_MISSED"})

    elif choice == "6":
        send("INJECT_VOICE", {"action": "APPLIANCE_ON"})

    elif choice == "7":
        send("INJECT_VOICE", {"action": "APPLIANCE_OFF"})

    elif choice == "8":
        send("INJECT_VOICE", {"action": "READ_SCHEDULE"})

    elif choice == "9":
        send("INJECT_VOICE", {"action": "REPEAT_LAST"})

    elif choice == "10":
        send("TRIGGER_SOS_BUTTON")

    elif choice == "11":
        body = _build_mock_sms_payload(
            drug_name="Atenolol",
            dosage="50mg",
            time_due="07:30",
        )
        send("INJECT_INBOUND_SMS", {
            "sender": "+233200510903",
            "body":   body,
        })

    elif choice == "12":
        body = _build_mock_sms_payload(
            drug_name="Atenolol",
            dosage="50mg",
            time_due="07:30",
        )
        send("INJECT_INBOUND_SMS", {
            "sender": "+233200000000",   # not a registered caregiver
            "body":   body,
        })

    elif choice == "13":
        send("INJECT_INBOUND_SMS", {
            "sender": "+233200510903",
            "body":   "MED|INSERT|broken",   # missing fields
        })

    elif choice == "14":
        action = input("    Twi phrase OR action constant: ").strip()
        if action:
            send("INJECT_VOICE", {"action": action})

    elif choice == "15":
        sender = input("    Sender phone: ").strip()
        body   = input("    SMS body:     ").strip()
        if sender and body:
            send("INJECT_INBOUND_SMS", {"sender": sender, "body": body})

    else:
        print(f"  Unknown choice: {choice!r}")

    return True


# ===========================================================================
# Main loop
# ===========================================================================

def main() -> int:
    print()
    print(BANNER_RULE)
    print("  GERIATRIC HUB — DEVELOPER CONSOLE")
    print("  Drives the running hub via the DevCommandQueue IPC channel.")
    print(BANNER_RULE)

    if not hub_is_running():
        print("\n  ⚠  No response from hub. Make sure `python -m src.main` is")
        print("     running in another terminal before continuing.")
        print("     Console will still queue commands — they will apply if/when")
        print("     the hub starts.")
    else:
        print("\n  ✓  Hub is responding. Ready to send commands.")

    while True:
        print(MENU)
        try:
            choice = input("  Choice: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not handle_choice(choice):
            break

    print("\n  Goodbye.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
    