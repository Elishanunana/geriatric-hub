"""
dev_command_poller.py
=====================
Debug-only IPC bridge between the `dev_console` process and the running
hub process.

The console (a separate Python process running in another terminal)
inserts rows into the DevCommandQueue table. This poller, which runs as
a background thread inside the hub, drains the queue every second and
applies each command against the hub's live in-memory mock instances.

This file is part of the development tooling and is not used in
production. The DevCommandQueue table is created by db_init.py but is
read by no other subsystem.

Supported Commands
------------------
    TOGGLE_WIFI         args: {"connected": bool}
    INJECT_VOICE        args: {"action": str, "twi_phrase": str (optional)}
    INJECT_INBOUND_SMS  args: {"sender": str, "body": str}
    TRIGGER_SOS_BUTTON  args: {} (no args needed)
    PING                args: {} (sanity check — produces "pong" result)

Author: Wise (Asumang Pobi Godwin) — KNUST COE 497
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from src.data_management.db_connection import get_connection
from src.hardware_mocks.mock_microphone import (
    CommandEvent,
    TWI_VOCABULARY,
    ACTION_DOSE_CONFIRMED,
    ACTION_DOSE_MISSED,
    ACTION_SOS,
    ACTION_APPLIANCE_ON,
    ACTION_APPLIANCE_OFF,
    ACTION_READ_SCHEDULE,
    ACTION_REPEAT_LAST,
)

logger = logging.getLogger(__name__)


DEFAULT_POLL_INTERVAL_SECONDS = 1.0


class DevCommandPoller:
    """
    Drains DevCommandQueue and applies commands to the hub's mocks.
    Threading-safe. Receives direct references to the live mock
    instances during construction so it can call methods on them.
    """

    def __init__(
        self,
        *,
        microphone: Any,
        gsm: Any,
        gpio: Any,
        arbiter: Any,
        voice_fanout: Any,
        poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    ):
        self._microphone   = microphone
        self._gsm          = gsm
        self._gpio         = gpio
        self._arbiter      = arbiter
        self._voice_fanout = voice_fanout
        self._poll_interval = poll_interval_seconds

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="DevCommandPoller", daemon=True,
        )
        self._thread.start()
        logger.info(
            "DevCommandPoller started (DEBUG IPC — poll = %ss).",
            self._poll_interval,
        )

    def stop(self, join_timeout: float = 3.0) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=join_timeout)
        logger.info("DevCommandPoller stopped.")

    # -----------------------------------------------------------------------
    # Polling loop
    # -----------------------------------------------------------------------

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._drain_once()
            except Exception:
                logger.exception("DevCommandPoller drain raised; continuing.")
            self._stop_event.wait(timeout=self._poll_interval)

    def _drain_once(self) -> None:
        """Process every unapplied command in FIFO order."""
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT cmd_id, command, args_json
                  FROM DevCommandQueue
                 WHERE applied_at IS NULL
                 ORDER BY cmd_id ASC;
                """
            ).fetchall()

        for row in rows:
            self._apply_one(row["cmd_id"], row["command"], row["args_json"])

    def _apply_one(self, cmd_id: int, command: str, args_json: Optional[str]) -> None:
        try:
            args = json.loads(args_json) if args_json else {}
        except json.JSONDecodeError:
            self._mark_applied(cmd_id, "error: invalid args_json")
            return

        try:
            result = self._dispatch(command, args)
        except Exception as exc:
            logger.exception("DevCommand %s failed", command)
            self._mark_applied(cmd_id, f"error: {exc}")
            return

        self._mark_applied(cmd_id, result)

    # -----------------------------------------------------------------------
    # Command dispatch
    # -----------------------------------------------------------------------

    def _dispatch(self, command: str, args: dict) -> str:
        if command == "TOGGLE_WIFI":
            connected = bool(args.get("connected", True))
            self._arbiter.set_wifi_caregiver_connected(connected)
            return f"wifi_connected={connected}"

        if command == "INJECT_VOICE":
            action = args.get("action") or args.get("twi_phrase")
            if not action:
                return "error: INJECT_VOICE requires 'action' or 'twi_phrase'"
            event = self._microphone.inject_command(action)
            if event is None:
                return f"error: action '{action}' not recognized"
            # Fan-out also runs because the microphone's callback IS the
            # fan-out — inject_command triggers it internally. No extra
            # work needed here.
            return f"injected: {event.action}"

        if command == "INJECT_INBOUND_SMS":
            sender = args.get("sender")
            body   = args.get("body")
            if not sender or not body:
                return "error: INJECT_INBOUND_SMS requires 'sender' and 'body'"
            msg = self._gsm.inject_inbound_sms(sender, body)
            return f"injected at index={msg.index}"

        if command == "TRIGGER_SOS_BUTTON":
            self._gpio.trigger_sos()
            return "ok"

        if command == "PING":
            return "pong"

        return f"error: unknown command {command!r}"

    # -----------------------------------------------------------------------
    # Bookkeeping
    # -----------------------------------------------------------------------

    @staticmethod
    def _mark_applied(cmd_id: int, result: str) -> None:
        ts = datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z")
        with get_connection() as conn:
            conn.execute(
                """
                UPDATE DevCommandQueue
                   SET applied_at = ?, result = ?
                 WHERE cmd_id = ?;
                """,
                (ts, result, cmd_id),
            )
            