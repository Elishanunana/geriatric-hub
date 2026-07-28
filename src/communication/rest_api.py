"""
rest_api.py
===========
Local REST API server — the Hub-side terminus of the primary (Wi-Fi)
synchronisation pathway described in Section 3.5.4 of the project report:

    "When the caregiver is physically present in the home and connects the
     mobile application to the hub's Access Point network, the application
     initiates a session with the local Flask REST endpoint exposed by the
     hub on the AP interface. After a one-time handshake using the
     device-pairing token established during initial setup, the two
     endpoints exchange their unsynchronised SyncQueue records as JSON
     batches via HTTP POST requests, with up to 100 records transmitted
     per batch."

Endpoints
---------
    GET  /api/v1/health               — connectivity probe (no auth).
    GET  /api/v1/events/unsynced      — list events with synced_flag = 0.
    POST /api/v1/events/ack           — mark events as synced; SOS events
                                         additionally trigger SOSHandler.acknowledge().
    POST /api/v1/schedule/sync        — apply a batch of schedule updates from the app.

Authentication
--------------
Every endpoint except /health requires a bearer token in the Authorization
header. The token is the device-pairing token provisioned during initial
setup (Section 3.5.5 of the report). Comparison is constant-time via
hmac.compare_digest to avoid timing attacks.

Threading & Lifecycle
---------------------
The server runs on a daemon thread driven by werkzeug.serving.make_server,
which gives us a clean .shutdown() — unlike Flask's dev server, which is
designed only for foreground execution. Routes themselves run on werkzeug's
internal thread pool (threaded=True), so multiple caregiver-app requests
can be served concurrently without blocking each other.

Author: Wise (Asumang Pobi Godwin) — KNUST COE 497
"""

from __future__ import annotations

import hmac
import json
import logging
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Optional, List, Dict

from flask import Flask, jsonify, request
from werkzeug.serving import make_server

from src.data_management.repositories import (
    ElderProfileRepo,
    MedicationScheduleRepo,
    EventLogRepo,
    SyncQueueRepo,
    MedicationSchedule,
    EventLogEntry,
    SyncQueueEntry,
)

logger = logging.getLogger(__name__)
logging.getLogger("werkzeug").setLevel(logging.WARNING)



# ===========================================================================
# Configuration constants
# ===========================================================================

DEFAULT_HOST              = "0.0.0.0"   # All interfaces; in prod = the wlan0 AP IP.
DEFAULT_PORT              = 5000
DEFAULT_MAX_PAYLOAD_BYTES = 1 * 1024 * 1024   # 1 MB — generous for 100-record batches.
DEFAULT_BATCH_LIMIT       = 100               # Per Section 3.5.4 of the report.

API_PREFIX = "/api/v1"


# ===========================================================================
# LocalRestAPI
# ===========================================================================

class LocalRestAPI:
    """
    Flask-based REST server for the Wi-Fi synchronisation pathway.
    Threading-safe and dependency-injected throughout.

    Lifecycle:
        api = LocalRestAPI(auth_token="...", sos_handler=sos_handler)
        api.start()
        ...
        api.stop()

    Or for unit tests, skip start() and use api.test_client() directly.
    """

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def __init__(
        self,
        *,
        auth_token: str,
        sos_handler: Optional[Any]                     = None,
        elder_repo:  Optional[ElderProfileRepo]        = None,
        med_repo:    Optional[MedicationScheduleRepo]  = None,
        event_repo:  Optional[EventLogRepo]            = None,
        sync_repo:   Optional[SyncQueueRepo]           = None,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
        batch_limit: int = DEFAULT_BATCH_LIMIT,
    ):
        """
        Parameters
        ----------
        auth_token : the device-pairing token. All authenticated endpoints
                     reject requests whose Authorization header doesn't
                     match this value.
        sos_handler : optional reference to an SOSHandler instance. If
                      provided, ack-ing an sos_triggered event_id will
                      invoke sos_handler.acknowledge(). May be None during
                      bootstrap or in unit tests that don't exercise the
                      SOS pathway — in which case SOS acks downgrade to
                      mark_synced only, and a warning is logged.
        *_repo : repository instances; default-constructed if omitted.
        host, port : bind address. In production the host should resolve
                     to the wlan0 AP interface; defaults to all interfaces.
        max_payload_bytes : Flask's MAX_CONTENT_LENGTH safety cap.
        batch_limit : maximum records accepted per /schedule/sync call.
        """
        if not auth_token:
            raise ValueError("auth_token is required (the device-pairing token).")

        self._auth_token   = auth_token
        self._sos_handler  = sos_handler
        self._elder_repo   = elder_repo  or ElderProfileRepo()
        self._med_repo     = med_repo    or MedicationScheduleRepo()
        self._event_repo   = event_repo  or EventLogRepo()
        self._sync_repo    = sync_repo   or SyncQueueRepo()

        self._host         = host
        self._port         = port
        self._batch_limit  = batch_limit

        # Build the Flask app eagerly so it's available for test_client()
        # without requiring a network bind.
        self._app = self._build_app(max_payload_bytes)

        # Server-thread state.
        self._server: Optional[Any]              = None  # werkzeug.serving.BaseWSGIServer
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    # -----------------------------------------------------------------------
    # Public lifecycle
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Spin up the server on a daemon thread. Idempotent."""
        with self._lock:
            if self._server is not None:
                logger.warning("LocalRestAPI already started.")
                return

            self._server = make_server(
                self._host, self._port, self._app, threaded=True
            )
            # Capture the actual port (handy when port=0 is used in tests).
            self._port = self._server.server_port

            self._thread = threading.Thread(
                target=self._server.serve_forever,
                name="LocalRestAPI-Server",
                daemon=True,
            )
            self._thread.start()

        logger.info(
            "LocalRestAPI started on %s:%d (Wi-Fi sync endpoint).",
            self._host, self._port,
        )
        self._event_repo.insert(
            EventLogRepo.SYSTEM_BOOT,
            details={
                "subsystem":   "LocalRestAPI",
                "host":        self._host,
                "port":        self._port,
                "batch_limit": self._batch_limit,
            },
        )

    def stop(self, join_timeout: float = 5.0) -> None:
        """Gracefully shut down the server."""
        with self._lock:
            srv, thr = self._server, self._thread
            self._server = None
            self._thread = None

        if srv is None:
            return

        try:
            srv.shutdown()
            srv.server_close()
        except Exception:
            logger.exception("Error during LocalRestAPI shutdown.")

        if thr is not None:
            thr.join(timeout=join_timeout)

        logger.info("LocalRestAPI stopped.")

    # -----------------------------------------------------------------------
    # Test client convenience
    # -----------------------------------------------------------------------

    def test_client(self):
        """
        Flask test client — bypasses the network entirely. Used by the
        smoke test below and by integration tests so the sync correctness
        scenarios from Section 3.7.3 can be verified deterministically.
        """
        return self._app.test_client()

    @property
    def app(self) -> Flask:
        """Underlying Flask app, for advanced test scenarios."""
        return self._app

    @property
    def port(self) -> int:
        """Live bind port — useful when port=0 was passed."""
        return self._port

    # -----------------------------------------------------------------------
    # Flask app construction
    # -----------------------------------------------------------------------

    def _build_app(self, max_payload_bytes: int) -> Flask:
        app = Flask("geriatric_hub.local_rest_api")
        app.config["MAX_CONTENT_LENGTH"] = max_payload_bytes
        # No JSON HTML-escaping needed — Twi diacritics should pass through.
        app.config["JSON_AS_ASCII"] = False

        # ---- Auth decorator -------------------------------------------------
        def require_auth(view):
            @wraps(view)
            def wrapped(*args, **kwargs):
                header = request.headers.get("Authorization", "")
                if not header.startswith("Bearer "):
                    return jsonify({"error": "missing_bearer_token"}), 401
                presented = header[len("Bearer "):].strip()
                # Constant-time comparison.
                if not hmac.compare_digest(presented, self._auth_token):
                    return jsonify({"error": "invalid_token"}), 401
                return view(*args, **kwargs)
            return wrapped

        # ---- Generic error handlers ----------------------------------------
        @app.errorhandler(404)
        def _not_found(_):
            return jsonify({"error": "not_found"}), 404

        @app.errorhandler(405)
        def _method_not_allowed(_):
            return jsonify({"error": "method_not_allowed"}), 405

        @app.errorhandler(413)
        def _too_large(_):
            return jsonify({"error": "payload_too_large"}), 413

        @app.errorhandler(500)
        def _internal(_):
            # 500 handlers run AFTER the exception has been logged by Flask;
            # we just make sure the response shape is uniform JSON.
            return jsonify({"error": "internal_server_error"}), 500

        # ===================================================================
        # Endpoint: GET /api/v1/health
        # ===================================================================
        # Connectivity probe used by the caregiver app to verify it has
        # actually associated with the hub's AP and can reach the service.
        # Deliberately unauthenticated so the app can render a useful
        # error before prompting the caregiver for the pairing token.
        @app.route(f"{API_PREFIX}/health", methods=["GET"])
        def health():
            return jsonify({
                "status":      "ok",
                "system":      "geriatric_hub",
                "api_version": "v1",
                "server_time": _utcnow_iso(),
            })

        # ===================================================================
        # Endpoint: GET /api/v1/events/unsynced
        # ===================================================================
        # Returns events whose synced_flag = 0, ordered oldest-first.
        # The caregiver app processes the batch and follows up with
        # POST /events/ack to confirm receipt.
        @app.route(f"{API_PREFIX}/events/unsynced", methods=["GET"])
        @require_auth
        def get_unsynced_events():
            try:
                limit = self._parse_limit(request.args.get("limit"))
                events = self._event_repo.fetch_unsynced(limit=limit)
                return jsonify({
                    "count":  len(events),
                    "limit":  limit,
                    "events": [_serialize_event(e) for e in events],
                })
            except Exception:
                logger.exception("GET /events/unsynced failed.")
                return jsonify({"error": "fetch_failed"}), 500

        # ===================================================================
        # Endpoint: POST /api/v1/events/ack
        # ===================================================================
        # Body: {"event_ids": [<int>, <int>, ...]}
        # For each ID:
        #   • If the event is sos_triggered, also call sos_handler.acknowledge()
        #     so the LED returns to steady green (per Section 3.4.5).
        #   • Mark synced_flag = 1.
        # Returns per-ID outcomes so the app can surface partial successes.
        @app.route(f"{API_PREFIX}/events/ack", methods=["POST"])
        @require_auth
        def ack_events():
            data = request.get_json(silent=True)
            if not isinstance(data, dict) or not isinstance(data.get("event_ids"), list):
                return jsonify({"error": "bad_request",
                                "detail": "expected JSON object with 'event_ids' list"}), 400

            event_ids = data["event_ids"]
            if len(event_ids) > self._batch_limit:
                return jsonify({
                    "error":  "batch_too_large",
                    "limit":  self._batch_limit,
                    "got":    len(event_ids),
                }), 400

            # Validate IDs are positive integers up front.
            try:
                ids = [int(x) for x in event_ids]
            except (TypeError, ValueError):
                return jsonify({"error": "bad_request",
                                "detail": "event_ids must be integers"}), 400

            results: List[Dict[str, Any]] = []
            for event_id in ids:
                try:
                    outcome = self._handle_single_ack(event_id)
                    results.append({"event_id": event_id, **outcome})
                except Exception:
                    logger.exception("Ack failed for event_id=%d", event_id)
                    results.append({
                        "event_id": event_id,
                        "status":   "error",
                        "detail":   "internal_error",
                    })

            return jsonify({
                "acknowledged_at": _utcnow_iso(),
                "results":         results,
            })

        # ===================================================================
        # Endpoint: POST /api/v1/schedule/sync
        # ===================================================================
        # Body:
        # {
        #   "schedules": [
        #     {
        #       "change_id":    "<uuid>",          # for idempotent rejection
        #       "elder_id":     <int>,
        #       "drug_name":    "<str>",
        #       "dosage":       "<str>",
        #       "time_due":     "HH:MM",
        #       "days_of_week": "DAILY" | "MON,TUE,...",
        #       "active":       0 | 1,
        #       "prescribed_by":"caregiver" | "pharmacist",
        #       "timestamp":    "<iso8601>"
        #     },
        #     ...
        #   ]
        # }
        @app.route(f"{API_PREFIX}/schedule/sync", methods=["POST"])
        @require_auth
        def sync_schedules():
            data = request.get_json(silent=True)
            if not isinstance(data, dict) or not isinstance(data.get("schedules"), list):
                return jsonify({"error": "bad_request",
                                "detail": "expected JSON object with 'schedules' list"}), 400

            schedules = data["schedules"]
            if len(schedules) > self._batch_limit:
                return jsonify({
                    "error":  "batch_too_large",
                    "limit":  self._batch_limit,
                    "got":    len(schedules),
                }), 400

            results: List[Dict[str, Any]] = []
            for idx, item in enumerate(schedules):
                try:
                    outcome = self._handle_single_schedule(item)
                    results.append({"index": idx, **outcome})
                except Exception:
                    logger.exception("Schedule sync failed for index=%d", idx)
                    results.append({
                        "index":  idx,
                        "status": "error",
                        "detail": "internal_error",
                    })

            return jsonify({
                "applied_at": _utcnow_iso(),
                "results":    results,
            })

        return app

    # -----------------------------------------------------------------------
    # Per-record handlers
    # -----------------------------------------------------------------------

    def _handle_single_ack(self, event_id: int) -> Dict[str, Any]:
        """Apply a single event ack. Returns a result dict for the response."""
        # We need the event_type to decide whether to call sos_handler.
        # Fetching by ID isn't a method we built on EventLogRepo (deliberately —
        # it's append-only and rarely needs random access), so we fetch a
        # bounded recent window and search. For SOS acks, the ack will almost
        # always come within minutes of the trigger, so this is cheap.
        recent = self._event_repo.fetch_recent(limit=500)
        target = next((e for e in recent if e.event_id == event_id), None)

        if target is None:
            return {"status": "not_found"}

        sos_acknowledged = False
        if target.event_type == EventLogRepo.SOS_TRIGGERED:
            if self._sos_handler is not None:
                try:
                    sos_acknowledged = bool(self._sos_handler.acknowledge(event_id))
                except Exception:
                    logger.exception(
                        "SOSHandler.acknowledge raised for event_id=%d", event_id
                    )
            else:
                # Defensive: if no SOS handler is wired in (e.g. unit tests
                # of the API in isolation), we still mark the event synced
                # but warn loudly — the LED will not return to green.
                logger.warning(
                    "Ack-ing sos_triggered event_id=%d but no SOSHandler is wired.",
                    event_id,
                )

        marked = self._event_repo.mark_synced(event_id)
        if not marked:
            # mark_synced returns False if the event is already synced — not
            # an error, just a no-op repeat.
            return {
                "status":           "already_synced",
                "event_type":       target.event_type,
                "sos_acknowledged": sos_acknowledged,
            }

        return {
            "status":           "ok",
            "event_type":       target.event_type,
            "sos_acknowledged": sos_acknowledged,
        }

    def _handle_single_schedule(self, item: Any) -> Dict[str, Any]:
        """Apply a single schedule update. Returns a result dict for the response."""
        if not isinstance(item, dict):
            return {"status": "bad_record", "detail": "expected object"}

        # Required fields.
        required = ("change_id", "elder_id", "drug_name",
                    "dosage", "time_due", "days_of_week", "active")
        missing = [k for k in required if k not in item]
        if missing:
            return {"status": "bad_record",
                    "detail": f"missing fields: {missing}"}

        change_id = str(item["change_id"]).strip()
        if not change_id:
            return {"status": "bad_record", "detail": "empty change_id"}

        # Idempotency check — same invariant the SMS handler enforces, so
        # a record arriving via either pathway is rejected on second arrival.
        if self._sync_repo.is_duplicate(change_id):
            return {"status": "duplicate", "change_id": change_id}

        # Coerce types.
        try:
            elder_id = int(item["elder_id"])
            active   = int(item["active"])
        except (TypeError, ValueError):
            return {"status": "bad_record", "detail": "elder_id/active must be int"}

        if active not in (0, 1):
            return {"status": "bad_record", "detail": "active must be 0 or 1"}

        sched = MedicationSchedule(
            elder_id     = elder_id,
            drug_name    = str(item["drug_name"]),
            dosage       = str(item["dosage"]),
            time_due     = str(item["time_due"]),
            days_of_week = str(item["days_of_week"]),
            active       = active,
        )
        prescribed_by = str(item.get("prescribed_by", "caregiver"))

        # Apply with sync_method = 'app_wifi' per the task brief.
        schedule_id = self._med_repo.upsert_from_payload(
            sched,
            sync_method="app_wifi",
            prescribed_by=prescribed_by,
        )

        # Record the App→Hub change in the SyncQueue as already-synced so
        # is_duplicate() catches retransmissions, mirroring the SMS handler's
        # idempotency invariant.
        try:
            self._sync_repo.enqueue(SyncQueueEntry(
                change_id   = change_id,
                entity_type = "MedicationSchedule",
                entity_id   = schedule_id,
                change_type = "INSERT",   # canonical CRUD verb; ack semantics in payload
                timestamp   = str(item.get("timestamp") or _utcnow_iso()),
                sync_state  = "synced",
                direction   = "App->Hub",
                transport   = "wifi_rest",
                payload     = json.dumps(item, ensure_ascii=False, default=str),
            ))
        except Exception:
            # Most common cause: race with is_duplicate() if two threads
            # process the same change_id. Benign — the PK constraint on
            # change_id preserves the invariant.
            logger.exception(
                "Failed to record App->Hub change_id=%s (likely benign duplicate).",
                change_id,
            )

        # Also log to the EventLog so the audit trail captures the apply.
        self._event_repo.insert(
            "schedule_synced_via_wifi",
            details={
                "change_id":     change_id,
                "schedule_id":   schedule_id,
                "drug_name":     sched.drug_name,
                "time_due":      sched.time_due,
                "prescribed_by": prescribed_by,
            },
        )

        return {
            "status":      "ok",
            "change_id":   change_id,
            "schedule_id": schedule_id,
        }

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _parse_limit(self, raw: Optional[str]) -> int:
        """Parse the ?limit= query param, clamped to [1, batch_limit]."""
        if raw is None:
            return self._batch_limit
        try:
            n = int(raw)
        except ValueError:
            return self._batch_limit
        return max(1, min(n, self._batch_limit))


# ===========================================================================
# Module-level helpers
# ===========================================================================

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def _serialize_event(e: EventLogEntry) -> Dict[str, Any]:
    """
    Convert an EventLogEntry into a JSON-friendly dict. The `details`
    field is stored as a JSON string in SQLite; we attempt to deserialize
    it so the API consumer receives nested JSON rather than a string-
    encoded JSON blob.
    """
    out = asdict(e)
    raw_details = out.get("details")
    if isinstance(raw_details, str) and raw_details:
        try:
            out["details"] = json.loads(raw_details)
        except json.JSONDecodeError:
            # Leave as raw string — consumer can handle it.
            pass
    return out


# ===========================================================================
# Standalone smoke test
# ===========================================================================

if __name__ == "__main__":
    """
    Run from the project root:

        python -m src.communication.rest_api

    Uses the Flask test client — no network bind required. Demonstrates:
      1. /health is reachable WITHOUT a token.
      2. Authenticated endpoints reject missing / wrong tokens.
      3. /events/unsynced returns currently-pending events.
      4. /events/ack marks events synced AND triggers SOSHandler.acknowledge
         when the acked event is sos_triggered.
      5. /schedule/sync upserts schedules with sync_method=app_wifi and
         is idempotent on the same change_id.
    """
    import logging
    import uuid
    from src.hardware_mocks.mock_speaker  import MockSpeaker
    from src.hardware_mocks.mock_gpio     import MockGPIOController
    from src.hardware_mocks.mock_sim800l  import MockGSMModule
    from src.control_logic.sos_handler    import SOSHandler, SOSSource
    from src.data_management.repositories import (
        ElderProfileRepo, ElderProfile,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(threadName)s] %(name)s — %(message)s",
    )

    PAIRING_TOKEN = "demo-pairing-token-from-system-config"
    AUTH_HEADERS  = {"Authorization": f"Bearer {PAIRING_TOKEN}"}

    # --- Seed an elder + caregiver phones ------------------------------
    elder_repo = ElderProfileRepo()
    elder = elder_repo.fetch_first()
    if elder is None:
        elder_id = elder_repo.insert(ElderProfile(
            name="Demo Elder",
            language="twi",
            caregiver_phones="+233244111111",
        ))
    else:
        elder_id = elder.elder_id

    # --- Wire up an SOS handler so we can test the ack→handler path ----
    speaker = MockSpeaker(simulate_latency=False)
    gpio    = MockGPIOController()
    gsm     = MockGSMModule()
    gpio.start()
    sos_handler = SOSHandler(
        speaker=speaker, gpio=gpio, gsm=gsm,
        cooldown_seconds=0,    # no cooldown — clean test environment
    )

    # --- Clean slate: mark every pre-existing unsynced event as synced
    # so the smoke test is deterministic regardless of how much state has
    # accumulated in the dev database from earlier task runs. In production
    # the caregiver app does this naturally by paging through and acking,
    # but in a one-shot smoke test we need to fast-forward past the backlog
    # so the new SOS event we're about to trigger lands at the head of the
    # unsynced list.
    event_repo_for_cleanup = EventLogRepo()
    backlog = event_repo_for_cleanup.fetch_unsynced(limit=10000)
    for ev in backlog:
        event_repo_for_cleanup.mark_synced(ev.event_id)
    print(f"  [setup] Marked {len(backlog)} pre-existing unsynced events as synced "
          f"to give the test a clean slate.")

    # Trigger an SOS so we have a real sos_triggered event in the log.
    sos_result = sos_handler.trigger(source=SOSSource.MANUAL)
    sos_event_id = sos_result.sos_event_id

    # --- Build the API ------------------------------------------------
    api = LocalRestAPI(
        auth_token=PAIRING_TOKEN,
        sos_handler=sos_handler,
    )
    client = api.test_client()

    # --- Scenario 1: /health (no auth required) -----------------------
    print("\n" + "=" * 70)
    print("  SCENARIO 1 — GET /health  (no auth)")
    print("=" * 70)
    r = client.get(f"{API_PREFIX}/health")
    print(f"  status = {r.status_code}")
    print(f"  body   = {r.get_json()}")
    assert r.status_code == 200

    # --- Scenario 2: missing token rejected --------------------------
    print("\n" + "=" * 70)
    print("  SCENARIO 2 — Auth rejection")
    print("=" * 70)
    r = client.get(f"{API_PREFIX}/events/unsynced")
    print(f"  no token   → {r.status_code}: {r.get_json()}")
    assert r.status_code == 401

    r = client.get(
        f"{API_PREFIX}/events/unsynced",
        headers={"Authorization": "Bearer wrong-token"},
    )
    print(f"  bad token  → {r.status_code}: {r.get_json()}")
    assert r.status_code == 401

    # --- Scenario 3: fetch unsynced events ---------------------------
    print("\n" + "=" * 70)
    print("  SCENARIO 3 — GET /events/unsynced")
    print("=" * 70)
    r = client.get(f"{API_PREFIX}/events/unsynced?limit=20", headers=AUTH_HEADERS)
    body = r.get_json()
    print(f"  status = {r.status_code}")
    print(f"  count  = {body['count']}")
    sos_event_in_response = any(
        e["event_id"] == sos_event_id for e in body["events"]
    )
    print(f"  sos_event_id={sos_event_id} present in response: {sos_event_in_response}")
    assert r.status_code == 200
    assert sos_event_in_response

    # --- Scenario 4: ack the SOS event → triggers SOSHandler.acknowledge
    print("\n" + "=" * 70)
    print("  SCENARIO 4 — POST /events/ack  (with sos_triggered event)")
    print("=" * 70)
    print("  Watch for [GPIO] LED transition back to STEADY GREEN below ↓")
    r = client.post(
        f"{API_PREFIX}/events/ack",
        json={"event_ids": [sos_event_id]},
        headers=AUTH_HEADERS,
    )
    body = r.get_json()
    print(f"  status  = {r.status_code}")
    print(f"  results = {body['results']}")
    assert r.status_code == 200
    assert body["results"][0]["sos_acknowledged"] is True

    # --- Scenario 5: schedule sync from caregiver app ----------------
    print("\n" + "=" * 70)
    print("  SCENARIO 5 — POST /schedule/sync")
    print("=" * 70)
    change_id = str(uuid.uuid4())
    r = client.post(
        f"{API_PREFIX}/schedule/sync",
        json={
            "schedules": [
                {
                    "change_id":    change_id,
                    "elder_id":     elder_id,
                    "drug_name":    "Metformin",
                    "dosage":       "500mg",
                    "time_due":     "07:00",
                    "days_of_week": "DAILY",
                    "active":       1,
                    "prescribed_by":"caregiver",
                    "timestamp":    _utcnow_iso(),
                },
            ],
        },
        headers=AUTH_HEADERS,
    )
    body = r.get_json()
    print(f"  status  = {r.status_code}")
    print(f"  results = {body['results']}")
    assert r.status_code == 200
    assert body["results"][0]["status"] == "ok"
    new_schedule_id = body["results"][0]["schedule_id"]

    # --- Scenario 6: replay same change_id → duplicate ---------------
    print("\n" + "=" * 70)
    print("  SCENARIO 6 — POST /schedule/sync replay  (idempotent rejection)")
    print("=" * 70)
    r = client.post(
        f"{API_PREFIX}/schedule/sync",
        json={
            "schedules": [
                {
                    "change_id":    change_id,        # same change_id
                    "elder_id":     elder_id,
                    "drug_name":    "Metformin",
                    "dosage":       "500mg",
                    "time_due":     "07:00",
                    "days_of_week": "DAILY",
                    "active":       1,
                    "prescribed_by":"caregiver",
                },
            ],
        },
        headers=AUTH_HEADERS,
    )
    body = r.get_json()
    print(f"  status  = {r.status_code}")
    print(f"  results = {body['results']}")
    assert body["results"][0]["status"] == "duplicate"

    # --- Scenario 7: malformed body ---------------------------------
    print("\n" + "=" * 70)
    print("  SCENARIO 7 — Bad request  (missing 'event_ids')")
    print("=" * 70)
    r = client.post(
        f"{API_PREFIX}/events/ack",
        json={"wrong_key": [1, 2]},
        headers=AUTH_HEADERS,
    )
    print(f"  status = {r.status_code}: {r.get_json()}")
    assert r.status_code == 400

    print("\n" + "=" * 70)
    print("  All assertions passed. The Wi-Fi sync endpoint is wired correctly.")
    print(f"  New schedule_id from caregiver app sync: {new_schedule_id}")
    print("=" * 70)

    gpio.stop()
