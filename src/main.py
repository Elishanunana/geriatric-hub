"""
main.py
=======
System orchestrator — the single entry point that constructs every
subsystem of the Resilient, Offline-First Assistive Ecosystem and runs
them as one cohesive process.

This file is the operational realization of Section 3.5 of the project
report: it brings together the four software layers (Voice Engine, Control
Logic, Data Management, Communication) and starts them in the order
required for the system to function as a single hub.

Boot Sequence
-------------
The startup ordering is not arbitrary; it reflects dependency direction:

    1. Configuration       — load secrets and runtime knobs via ConfigService.
    2. Repositories        — shared instances; everything else takes them as deps.
    3. Hardware adapters   — speaker, mic, GPIO, GSM (mocks today, real drivers tomorrow).
    4. Communication layer — SMSTransport, Arbiter, SyncEngine, LocalRestAPI.
    5. Control Logic       — ReminderScheduler, SOSHandler, CommandDispatcher,
                             SMSPayloadHandler.
    6. Cross-wiring        — install the microphone fan-out, link the REST API
                             to the live SOSHandler, register GPIO button.
    7. Background threads  — start in dependency order: GPIO → REST API →
                             Scheduler → SMS Handler → Sync Engine → Microphone.
    8. Run loop            — main thread blocks until SIGINT, then graceful stop.

Configuration
-------------
All runtime parameters — secrets, network settings, polling intervals —
load through src.utils.config_service.ConfigService. The service produces
an immutable snapshot at boot; no subsystem can mutate it. To rotate
secrets (e.g., pairing-flow completion), update SystemConfig and restart
the hub.

Microphone Fan-Out
------------------
The keyword spotter publishes CommandEvents to a SINGLE callback. Three
control-logic subsystems need to consume those events: ReminderScheduler
(for DOSE_CONFIRMED / DOSE_MISSED), SOSHandler (for SOS), and
CommandDispatcher (for everything else). Rather than letting the last one
to wire itself overwrite the others (which is what happens in the
individual smoke tests), main.py installs a small fan-out dispatcher that
delivers each event to all three subscribers, with exception isolation so
a faulty subscriber cannot starve later ones in the chain.

Author: Wise (Asumang Pobi Godwin) — KNUST COE 497
"""

from __future__ import annotations

import logging
import os
import signal
import sys
import threading
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional

# -- Configuration -----------------------------------------------------------
from src.utils.config_service import ConfigService

# -- Data Management Layer ---------------------------------------------------
from src.data_management.db_init       import initialize_database
from src.data_management.repositories  import (
    ElderProfileRepo,
    MedicationScheduleRepo,
    EventLogRepo,
    SyncQueueRepo,
)

# -- Hardware Mocks ----------------------------------------------------------
from src.voice_engine.wav_speaker import WavSpeaker as MockSpeaker
from src.hardware_mocks.mock_microphone import CommandEvent
from src.voice_engine.whisper_spotter import WhisperSpotter as MockMicrophone
from src.control_logic.gpio_controller import GPIOController as MockGPIOController
from src.communication.sim800l_gsm import SIM800LGSMModule as MockGSMModule

# -- Control Logic Layer -----------------------------------------------------
from src.control_logic.reminder_scheduler   import ReminderScheduler
from src.control_logic.sos_handler          import SOSHandler
from src.control_logic.command_dispatcher   import CommandDispatcher
from src.control_logic.sms_payload_handler  import SMSPayloadHandler
from src.control_logic.dev_command_poller   import DevCommandPoller

# -- Communication Layer -----------------------------------------------------
from src.communication.sms_transport         import SMSTransport
from src.communication.connectivity_arbiter  import ConnectivityArbiter
from src.communication.sync_engine           import SyncEngine
from src.communication.rest_api              import LocalRestAPI


# ===========================================================================
# Logging configuration
# ===========================================================================

LOG_FORMAT  = "%(asctime)s [%(threadName)s] %(name)s — %(message)s"
LOG_LEVEL   = logging.INFO
BANNER_RULE = "═" * 70

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("hub.main")


# ===========================================================================
# Demo-mode override
# ===========================================================================
# Setting this to True forces dev-mock HMAC mode regardless of what
# SystemConfig says. Used during the academic defense demo so the
# caregiver-app side doesn't need to be plumbed with a matching key.
# Production images should leave this False and let SystemConfig drive.
#
# Flipped to False: the caregiver app now signs SMS payloads with the
# pairing token, and SystemConfig.hmac_key holds the same value, so the
# hub verifies real HMAC-SHA256 tags. Dev-console MOCK injections are
# consequently rejected at the HMAC stage (expected).
FORCE_DEV_MOCK_HMAC = False


# ===========================================================================
# Microphone fan-out dispatcher
# ===========================================================================

class VoiceEventFanOut:
    """
    Single callback that the microphone publishes to; internally fans the
    event out to all registered subscribers. Each subscriber's exceptions
    are caught and logged so a faulty downstream consumer cannot starve
    the SOS pathway (the most safety-critical subscriber).

    The order of subscription is preserved, but no subscriber is privileged
    — exception isolation makes ordering a non-issue for correctness.
    """

    def __init__(self):
        self._subscribers: List[Callable[[CommandEvent], None]] = []

    def subscribe(self, fn: Callable[[CommandEvent], None]) -> None:
        self._subscribers.append(fn)

    def __call__(self, event: CommandEvent) -> None:
        """Delivered to every subscriber; faulty ones are isolated."""
        for fn in self._subscribers:
            try:
                fn(event)
            except Exception:
                logger.exception(
                    "Voice-event subscriber %r raised on action=%s — isolating.",
                    getattr(fn, "__qualname__", fn), event.action,
                )


# ===========================================================================
# Hub orchestrator
# ===========================================================================

class Hub:
    """
    Owns every subsystem. start() brings the system online; stop() takes
    it down cleanly. Designed so the main() function is a thin shell
    around an instance of this class — easier to test and easier to extend.
    """

    def __init__(self):
        self.config: Optional[ConfigService] = None

        # Repositories.
        self.elder_repo: Optional[ElderProfileRepo]       = None
        self.med_repo:   Optional[MedicationScheduleRepo] = None
        self.event_repo: Optional[EventLogRepo]           = None
        self.sync_repo:  Optional[SyncQueueRepo]          = None

        # Hardware adapters.
        self.speaker:    Optional[MockSpeaker]         = None
        self.microphone: Optional[MockMicrophone]      = None
        self.gpio:       Optional[MockGPIOController]  = None
        self.gsm:        Optional[MockGSMModule]       = None

        # Communication layer.
        self.sms_transport: Optional[SMSTransport]        = None
        self.arbiter:       Optional[ConnectivityArbiter] = None
        self.sync_engine:   Optional[SyncEngine]          = None
        self.rest_api:      Optional[LocalRestAPI]        = None

        # Control logic.
        self.reminder_scheduler:  Optional[ReminderScheduler]   = None
        self.sos_handler:         Optional[SOSHandler]          = None
        self.command_dispatcher:  Optional[CommandDispatcher]   = None
        self.sms_payload_handler: Optional[SMSPayloadHandler]   = None
        # Debug-only IPC poller for the dev_console.
        self.dev_command_poller:  Optional[DevCommandPoller]    = None

        # Voice fan-out.
        self.voice_fanout: Optional[VoiceEventFanOut] = None

        # Stop flag for the run loop.
        self._stop = threading.Event()

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Run the full eight-step boot sequence."""
        self._banner("RESILIENT GERIATRIC CARE HUB — STARTING")

        # Step 1 — configuration.
        self._step(1, "Loading configuration via ConfigService")
        self._step_1_config()

        # Step 2 — repositories.
        self._step(2, "Instantiating shared repositories")
        self._step_2_repositories()

        # Step 3 — hardware adapters.
        self._step(3, "Instantiating hardware adapters (mock phase)")
        self._step_3_hardware()

        # Step 4 — communication layer.
        self._step(4, "Building communication layer")
        self._step_4_communication()

        # Step 5 — control logic.
        self._step(5, "Building control logic")
        self._step_5_control_logic()

        # Step 6 — cross-wiring.
        self._step(6, "Cross-wiring subsystems (mic fan-out, REST↔SOS, GPIO button)")
        self._step_6_cross_wire()

        # Step 7 — start background threads.
        self._step(7, "Starting background threads")
        self._step_7_start_threads()

        # Audit the system_boot event so the EventLog has a definitive
        # marker for forensics and for the panel demo.
        self.event_repo.insert(
            EventLogRepo.SYSTEM_BOOT,
            details={
                "subsystems": [
                    "ReminderScheduler", "SOSHandler", "CommandDispatcher",
                    "SMSPayloadHandler", "SMSTransport", "ConnectivityArbiter",
                    "SyncEngine", "LocalRestAPI", "MockMicrophone",
                ],
                "api_port":           self.config.api_port,
                "use_real_hmac":      self.config.use_real_hmac,
                "is_dev_mode":        self.config.is_dev_mode,
                "boot_completed_at":  _utcnow_iso(),
            },
        )

        self._banner("HUB ONLINE — Press Ctrl+C to shut down gracefully")

    def stop(self) -> None:
        """
        Graceful shutdown. Reverse dependency order: stop event-producing
        components first (microphone, SMS poll, sync engine), then the
        passive servers (REST API), then the orchestrators (scheduler),
        and finally the hardware adapters (GPIO clean release).
        """
        self._banner("HUB SHUTTING DOWN")
        self._stop.set()

        # Stop each subsystem inside its own try/except so one failure
        # cannot block the others — a hung GPIO release must not prevent
        # the REST server from closing its listening socket.
        for name, stop_fn in self._shutdown_steps():
            try:
                logger.info("Stopping %s...", name)
                stop_fn()
            except Exception:
                logger.exception("Error while stopping %s", name)

        try:
            self.event_repo.insert(
                "system_shutdown",
                details={"shutdown_at": _utcnow_iso()},
            )
        except Exception:
            logger.exception("Failed to write shutdown event to log.")

        self._banner("HUB OFFLINE — goodbye")

    def run_forever(self) -> None:
        """
        Block the main thread until SIGINT. All real work happens on the
        background threads we started in step 7; the main thread's only
        job is to absorb signals and keep the process alive.
        """
        # Install signal handlers — works on POSIX. On Windows, SIGINT is
        # raised by Ctrl+C even without explicit handler registration, so
        # we still get the KeyboardInterrupt in the run loop below.
        try:
            signal.signal(signal.SIGINT, self._on_signal)
            if hasattr(signal, "SIGTERM"):
                signal.signal(signal.SIGTERM, self._on_signal)
        except (ValueError, OSError):
            # signal.signal can fail if not running in the main thread of
            # the main interpreter. Falls back to KeyboardInterrupt only.
            logger.debug("Signal handlers not installed — relying on KeyboardInterrupt.")

        try:
            while not self._stop.is_set():
                # Long sleep with periodic wake — lets the stop_event
                # interrupt cleanly without busy-waiting.
                self._stop.wait(timeout=1.0)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received in run loop.")

    # -----------------------------------------------------------------------
    # Boot steps
    # -----------------------------------------------------------------------

    def _step_1_config(self) -> None:
        # Make sure the database exists and the schema is applied. This is
        # idempotent — db_init checks IF NOT EXISTS — so it's safe to run
        # on every boot.
        initialize_database(verbose=False)

        # Load the configuration via the ConfigService. Pass the demo-mode
        # override so dev runs don't depend on SystemConfig having
        # use_real_hmac=0; production images should set FORCE_DEV_MOCK_HMAC
        # to False and let SystemConfig drive.
        self.config = ConfigService.load(
            override_dev_mock_hmac=FORCE_DEV_MOCK_HMAC,
        )

        # Print the configuration summary. ConfigService is responsible
        # for not leaking secret values — its summary_lines() reports
        # placeholder/configured status without bytes.
        for line in self.config.summary_lines():
            logger.info(line)

    def _step_2_repositories(self) -> None:
        # SHARED instances — every consumer below uses these same handles
        # so all writes go through a consistent set of repository objects.
        # (At the SQLite layer each call still opens its own connection
        # via get_connection(), so concurrent threads stay isolated.)
        self.elder_repo = ElderProfileRepo()
        self.med_repo   = MedicationScheduleRepo()
        self.event_repo = EventLogRepo()
        self.sync_repo  = SyncQueueRepo()
        logger.info("  Repositories ready: ElderProfile, MedicationSchedule, EventLog, SyncQueue")

    def _step_3_hardware(self) -> None:
        self.speaker    = MockSpeaker(simulate_latency=False)
        self.microphone = MockMicrophone()
        self.gpio       = MockGPIOController()
        self.gsm        = MockGSMModule()
        logger.info("  Hardware adapters ready: Speaker, Microphone, GPIO, GSM")

    def _step_4_communication(self) -> None:
        # SMS transport — outbound side. The HMAC mode flows from the
        # ConfigService rather than a hard-coded module constant.
        self.sms_transport = SMSTransport(
            gsm           = self.gsm,
            elder_repo    = self.elder_repo,
            event_repo    = self.event_repo,
            sync_repo     = self.sync_repo,
            hmac_key      = self.config.hmac_key if self.config.use_real_hmac else "",
            dev_mock_hmac = not self.config.use_real_hmac,
        )

        # Arbiter — Wi-Fi state starts disconnected; the caregiver app's
        # association with the AP will flip it during normal operation.
        # In the mock phase, the dev console toggles it manually.
        self.arbiter = ConnectivityArbiter(
            sync_repo = self.sync_repo,
            gsm       = self.gsm,
            default_wifi_caregiver_connected = False,
        )

        # Sync engine — drives the outbound side of the SyncQueue.
        self.sync_engine = SyncEngine(
            sms_transport = self.sms_transport,
            arbiter       = self.arbiter,
            sync_repo     = self.sync_repo,
            event_repo    = self.event_repo,
            poll_interval_seconds = self.config.sync_poll_interval_seconds,
        )

        # REST API — sos_handler is wired in step 6 once the SOSHandler exists.
        self.rest_api = LocalRestAPI(
            auth_token  = self.config.pairing_token,
            sos_handler = None,                      # filled in step 6
            elder_repo  = self.elder_repo,
            med_repo    = self.med_repo,
            event_repo  = self.event_repo,
            sync_repo   = self.sync_repo,
            host        = self.config.api_host,
            port        = self.config.api_port,
        )
        logger.info("  Communication layer ready: SMSTransport, Arbiter, SyncEngine, REST API")

    def _step_5_control_logic(self) -> None:
        self.reminder_scheduler = ReminderScheduler(
            speaker    = self.speaker,
            microphone = self.microphone,
            med_repo   = self.med_repo,
            event_repo = self.event_repo,
            sync_repo  = self.sync_repo,
            poll_interval_seconds = self.config.reminder_poll_interval_seconds,
        )

        self.sos_handler = SOSHandler(
            speaker    = self.speaker,
            gpio       = self.gpio,
            gsm        = self.gsm,
            elder_repo = self.elder_repo,
            event_repo = self.event_repo,
            sync_repo  = self.sync_repo,
        )

        self.command_dispatcher = CommandDispatcher(
            speaker    = self.speaker,
            gpio       = self.gpio,
            med_repo   = self.med_repo,
            event_repo = self.event_repo,
        )

        self.sms_payload_handler = SMSPayloadHandler(
            gsm           = self.gsm,
            elder_repo    = self.elder_repo,
            med_repo      = self.med_repo,
            event_repo    = self.event_repo,
            sync_repo     = self.sync_repo,
            hmac_key      = self.config.hmac_key if self.config.use_real_hmac else "",
            dev_mock_hmac = not self.config.use_real_hmac,
            poll_interval_seconds = self.config.sms_poll_interval_seconds,
        )
        logger.info(
            "  Control logic ready: ReminderScheduler, SOSHandler, "
            "CommandDispatcher, SMSPayloadHandler"
        )

    def _step_6_cross_wire(self) -> None:
        # --- 6a. Microphone fan-out -------------------------------------
        self.voice_fanout = VoiceEventFanOut()
        self.voice_fanout.subscribe(self.reminder_scheduler.on_voice_command)
        self.voice_fanout.subscribe(self.sos_handler.on_voice_command)
        self.voice_fanout.subscribe(self.command_dispatcher.on_voice_command)
        self.microphone._on_command = self.voice_fanout
        logger.info("  Voice fan-out installed: Scheduler, SOSHandler, CommandDispatcher")

        # --- 6b. REST API → SOSHandler ---------------------------------
        self.rest_api._sos_handler = self.sos_handler
        logger.info("  LocalRestAPI cross-wired to live SOSHandler instance")

        # --- 6c. GPIO physical button -----------------------------------
        self.sos_handler.wire_to_gpio()
        logger.info("  GPIO physical button wired to SOSHandler")

        # --- 6d. Dev command poller (debug IPC for dev_console.py) -----
        # Deliberately NOT in production deployments — controlled by an
        # environment flag so the poller can be disabled when shipping.
        if os.environ.get("HUB_DEV_CONSOLE_ENABLED", "1") != "0":
            self.dev_command_poller = DevCommandPoller(
                microphone   = self.microphone,
                gsm          = self.gsm,
                gpio         = self.gpio,
                arbiter      = self.arbiter,
                voice_fanout = self.voice_fanout,
            )
            logger.info("  DevCommandPoller built (debug IPC channel for dev_console)")
        else:
            logger.info("  DevCommandPoller disabled via HUB_DEV_CONSOLE_ENABLED=0")

    def _step_7_start_threads(self) -> None:
        # Order: hardware → server → orchestrators → producers.
        # Producers go LAST so events have somewhere to land.
        self.gpio.start()
        logger.info("  ✓ GPIO controller started")

        self.rest_api.start()
        logger.info(
            "  ✓ Local REST API listening on %s:%d (Wi-Fi sync endpoint)",
            self.config.api_host, self.rest_api.port,
        )

        self.reminder_scheduler.start()
        logger.info("  ✓ Reminder scheduler running")

        self.sms_payload_handler.start()
        logger.info(
            "  ✓ SMS payload handler polling every %ds",
            self.config.sms_poll_interval_seconds,
        )

        self.sync_engine.start()
        logger.info(
            "  ✓ Sync engine running (poll = %ds)",
            self.config.sync_poll_interval_seconds,
        )

        # Microphone — once it starts emitting events, every downstream
        # consumer must already be online (which they are by this point).
        self.microphone.start()
        logger.info("  ✓ Microphone listening (terminal stdin in mock phase)")

        # Dev command poller — only if it was constructed in step 6d.
        # Started AFTER the microphone so that any voice commands injected
        # via the IPC channel are guaranteed to find an online fan-out.
        if self.dev_command_poller:
            self.dev_command_poller.start()
            logger.info("  ✓ DevCommandPoller running (debug IPC)")

    # -----------------------------------------------------------------------
    # Shutdown plan
    # -----------------------------------------------------------------------

    def _shutdown_steps(self):
        """
        Yield (name, callable) pairs in shutdown order. Each callable is
        wrapped in try/except by the caller so partial failures don't
        abort the rest of the teardown.
        """
        # Dev poller first — it's the source of test-injected commands;
        # silencing it before the mocks tear down prevents stray events
        # from racing the shutdown.
        if self.dev_command_poller:
            yield "DevCommandPoller", self.dev_command_poller.stop

        # Stop event PRODUCERS first so they stop emitting work.
        if self.microphone:
            yield "Microphone", self.microphone.stop
        if self.sms_payload_handler:
            yield "SMSPayloadHandler", self.sms_payload_handler.stop

        # Stop the orchestrator that drains the queue next.
        if self.sync_engine:
            yield "SyncEngine", self.sync_engine.stop

        # Stop the reminder scheduler — its dose workers may still be
        # mid-retry-sleep; .stop() wakes them via the shared stop_event.
        if self.reminder_scheduler:
            yield "ReminderScheduler", self.reminder_scheduler.stop

        # Stop the passive HTTP server.
        if self.rest_api:
            yield "LocalRestAPI", self.rest_api.stop

        # Hardware last — cleanest LED release.
        if self.gpio:
            yield "GPIO", self.gpio.stop

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _on_signal(self, signum, _frame) -> None:
        logger.info("Signal %d received — initiating shutdown.", signum)
        self._stop.set()

    @staticmethod
    def _step(num: int, label: str) -> None:
        logger.info("")
        logger.info("[STEP %d] %s", num, label)

    @staticmethod
    def _banner(text: str) -> None:
        # Print to stdout directly (not via logger) so the banner is
        # visually unmissable in the terminal.
        print()
        print(BANNER_RULE)
        print(f"  {text}")
        print(BANNER_RULE)
        print()


# ===========================================================================
# Module helpers
# ===========================================================================

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


# ===========================================================================
# Entry point
# ===========================================================================

def main() -> int:
    """
    Top-level entry. Returns a process exit code (0 = clean shutdown,
    non-zero = boot failure).
    """
    hub = Hub()

    try:
        hub.start()
    except Exception:
        logger.exception("Fatal error during boot — aborting.")
        try:
            hub.stop()
        except Exception:
            pass
        return 1

    try:
        hub.run_forever()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received at top level.")
    finally:
        hub.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
    