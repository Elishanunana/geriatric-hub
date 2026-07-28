"""
config_service.py
=================
Central configuration service. Single source of truth for every runtime
parameter the hub depends on — secrets (pairing token, HMAC key), network
settings (Wi-Fi AP SSID and PSK, REST host and port), and timing knobs
(poll intervals for the various background loops).

Why a dedicated service?
------------------------
Earlier iterations of main.py inlined a HubConfig helper that read a
handful of SystemConfig keys directly. That worked, but it concentrated
several concerns in main.py that don't belong there:

  • Secret-handling policy (placeholder detection, dev-vs-prod warnings).
  • SystemConfig schema knowledge (key names, default values, type coercion).
  • Validation logic (which keys are mandatory, which can fall back).

Moving these into a small dedicated service has three benefits:

  1. Future tooling (the device-pairing CLI, an admin web UI, the
     Flutter caregiver app's pairing flow) can reuse the same loader
     and validator without depending on main.py.

  2. Tests can construct a ConfigService from an in-memory dict instead
     of monkey-patching the database, making config-related tests fast
     and isolated.

  3. The use_real_hmac flag — previously a module-level constant in
     main.py — moves into SystemConfig where it belongs. Operators
     can flip it during the pairing flow without code changes.

Immutability
------------
ConfigService is a frozen dataclass. Once loaded at boot, no subsystem
can mutate it — even by accident. If a config value needs to change
(typically during the pairing flow), the hub must be restarted. This
matches the report's pairing model in Section 3.5.5 ("Initial pairing
between the application and the hub is performed once during system
setup") and prevents the subtle inconsistency bugs that arise when
secrets change mid-flight (e.g., an HMAC key swap while a payload is
being signed).

Author: Wise (Asumang Pobi Godwin) — KNUST COE 497
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.data_management.db_connection import get_connection

logger = logging.getLogger(__name__)


# ===========================================================================
# Placeholder values — warned about at load time
# ===========================================================================
# Any of these strings, if present, indicates the operator has not yet
# completed the pairing flow. The system runs in development mode but the
# warnings make this state visible in every boot log.

PLACEHOLDER_VALUES = frozenset({
    "CHANGE_ME_ON_PAIRING",
    "demo-pairing-token-from-system-config",
    "demo-hmac-key-from-system-config",
})


# ===========================================================================
# Safe development defaults
# ===========================================================================
# Used only when the corresponding key is missing entirely from SystemConfig
# (i.e., the database is brand-new and db_init.py's DEFAULT_CONFIG seeds
# haven't been applied). In normal operation the seeds provide
# 'CHANGE_ME_ON_PAIRING' for secrets, and these defaults never fire.

_DEFAULT_PAIRING_TOKEN          = "demo-pairing-token-from-system-config"
_DEFAULT_HMAC_KEY               = "demo-hmac-key-from-system-config"
_DEFAULT_AP_SSID                = "GeriatricHub"
_DEFAULT_AP_PSK                 = "CHANGE_ME_ON_PAIRING"
_DEFAULT_API_HOST               = "0.0.0.0"
_DEFAULT_API_PORT               = 5000
_DEFAULT_SMS_POLL_SECONDS       = 30
_DEFAULT_REMINDER_POLL_SECONDS  = 30
_DEFAULT_SYNC_POLL_SECONDS      = 30


# ===========================================================================
# ConfigService
# ===========================================================================

@dataclass(frozen=True)
class ConfigService:
    """
    Immutable snapshot of the hub's runtime configuration. Constructed
    once at boot via .load() and passed by reference to every subsystem
    that needs it.

    The frozen=True flag means any attempt to mutate a field after
    construction raises FrozenInstanceError — config drift is impossible.
    """

    # ---- Secrets ----------------------------------------------------------
    pairing_token: str
    hmac_key:      str

    # ---- Network ----------------------------------------------------------
    ap_ssid:       str
    ap_psk:        str
    api_host:      str
    api_port:      int

    # ---- Behaviour flags --------------------------------------------------
    # When True, the SMS pathway uses real HMAC-SHA256 verification.
    # When False, the dev-mock 'HMAC=MOCK' bypass is accepted. Production
    # deployments MUST set this to True via the pairing flow.
    use_real_hmac: bool

    # ---- Timing -----------------------------------------------------------
    sms_poll_interval_seconds:      int
    reminder_poll_interval_seconds: int
    sync_poll_interval_seconds:     int

    # ---- Computed flags exposed for convenience ---------------------------
    # Whether the secrets currently in use are placeholders. Subsystems
    # CAN read this if they want to gate behaviour (e.g., a future admin
    # endpoint that refuses to operate while in dev mode).
    is_dev_mode: bool = field(default=False)

    # =======================================================================
    # Loader
    # =======================================================================

    @classmethod
    def load(cls, *, override_dev_mock_hmac: Optional[bool] = None) -> "ConfigService":
        """
        Read every SystemConfig row, fold in safe defaults for any missing
        keys, validate the result, and return an immutable snapshot.

        Parameters
        ----------
        override_dev_mock_hmac : bool, optional
            For tests and one-off invocations: force the use_real_hmac
            flag regardless of what the database says. Pass `False` to
            force mock-HMAC mode (matches the prior main.py default for
            the demo). Pass `None` (default) to honour the database value.
        """
        rows = cls._read_system_config()

        # ---- Coercion with safe defaults ----------------------------------
        pairing_token = rows.get("pairing_token") or _DEFAULT_PAIRING_TOKEN
        hmac_key      = rows.get("hmac_key")      or _DEFAULT_HMAC_KEY
        ap_ssid       = rows.get("ap_ssid")       or _DEFAULT_AP_SSID
        ap_psk        = rows.get("ap_psk")        or _DEFAULT_AP_PSK
        api_host      = rows.get("api_host")      or _DEFAULT_API_HOST

        api_port = cls._coerce_int(
            rows.get("api_port"), _DEFAULT_API_PORT, "api_port",
        )
        sms_poll = cls._coerce_int(
            rows.get("sms_poll_interval_seconds"),
            _DEFAULT_SMS_POLL_SECONDS, "sms_poll_interval_seconds",
        )
        reminder_poll = cls._coerce_int(
            rows.get("reminder_poll_interval_seconds"),
            _DEFAULT_REMINDER_POLL_SECONDS, "reminder_poll_interval_seconds",
        )
        sync_poll = cls._coerce_int(
            rows.get("sync_poll_interval_seconds"),
            _DEFAULT_SYNC_POLL_SECONDS, "sync_poll_interval_seconds",
        )

        # use_real_hmac is stored as the string '0' or '1' in the DB.
        # Override (if provided) takes precedence.
        if override_dev_mock_hmac is not None:
            use_real_hmac = not override_dev_mock_hmac
        else:
            use_real_hmac = rows.get("use_real_hmac", "0") == "1"

        # is_dev_mode is computed from the secrets, not stored.
        is_dev_mode = (
            pairing_token in PLACEHOLDER_VALUES
            or hmac_key   in PLACEHOLDER_VALUES
        )

        cfg = cls(
            pairing_token                   = pairing_token,
            hmac_key                        = hmac_key,
            ap_ssid                         = ap_ssid,
            ap_psk                          = ap_psk,
            api_host                        = api_host,
            api_port                        = api_port,
            use_real_hmac                   = use_real_hmac,
            sms_poll_interval_seconds       = sms_poll,
            reminder_poll_interval_seconds  = reminder_poll,
            sync_poll_interval_seconds      = sync_poll,
            is_dev_mode                     = is_dev_mode,
        )
        cfg._validate_and_warn()
        return cfg

    # =======================================================================
    # Validation
    # =======================================================================

    def _validate_and_warn(self) -> None:
        """
        Emit log warnings for any condition that's safe but indicates the
        system isn't yet in a deployment-ready state. We never raise from
        here — placeholders are normal during development and during the
        period between pairing-flow steps. The warnings make the state
        observable.
        """
        # Secret placeholders.
        if self.pairing_token in PLACEHOLDER_VALUES:
            logger.warning(
                "pairing_token is a development placeholder — rotate via "
                "the pairing flow before any real deployment."
            )
        if self.hmac_key in PLACEHOLDER_VALUES:
            logger.warning(
                "hmac_key is a development placeholder — rotate via the "
                "pairing flow before any real deployment."
            )
        if self.ap_psk in PLACEHOLDER_VALUES:
            logger.warning(
                "ap_psk is a development placeholder — rotate before any "
                "real deployment so the AP doesn't broadcast a known key."
            )

        # Combination warning: real HMAC + placeholder key would silently
        # break inbound SMS validation, since the caregiver app would have
        # signed with the real key and the hub would verify with a placeholder.
        if self.use_real_hmac and self.hmac_key in PLACEHOLDER_VALUES:
            logger.warning(
                "use_real_hmac is TRUE but hmac_key is still a placeholder — "
                "all inbound SMS payloads will fail HMAC verification until "
                "the key is rotated."
            )

        # Sanity check on the API port.
        if not (1 <= self.api_port <= 65535):
            logger.error(
                "api_port=%d is outside the valid TCP port range — REST API "
                "will fail to bind.", self.api_port,
            )

    # =======================================================================
    # Logging summary — used by main.py at boot
    # =======================================================================

    def summary_lines(self) -> list[str]:
        """
        Produce a list of one-line strings describing the active config.
        Secrets are NEVER included in full — only their placeholder/real
        status is reported, so the boot log is safe to share verbatim.
        """
        return [
            f"  pairing_token : {self._secret_status(self.pairing_token)}",
            f"  hmac_key      : {self._secret_status(self.hmac_key)}",
            f"  use_real_hmac : {self.use_real_hmac}",
            f"  ap_ssid       : {self.ap_ssid}",
            f"  ap_psk        : {self._secret_status(self.ap_psk)}",
            f"  api           : {self.api_host}:{self.api_port}",
            f"  sms_poll      : {self.sms_poll_interval_seconds}s",
            f"  reminder_poll : {self.reminder_poll_interval_seconds}s",
            f"  sync_poll     : {self.sync_poll_interval_seconds}s",
            f"  dev_mode      : {self.is_dev_mode}",
        ]

    @staticmethod
    def _secret_status(value: str) -> str:
        """Report a secret's status without revealing its bytes."""
        if value in PLACEHOLDER_VALUES:
            return "<placeholder — rotate before deployment>"
        return f"<configured, length={len(value)}>"

    # =======================================================================
    # Internals
    # =======================================================================

    @staticmethod
    def _read_system_config() -> Dict[str, str]:
        """
        Read every row of SystemConfig into a flat dict. Done in a single
        SELECT to minimise DB round-trips — the table is small (typically
        under 20 rows) and we want every value at once.
        """
        result: Dict[str, str] = {}
        try:
            with get_connection() as conn:
                rows = conn.execute(
                    "SELECT config_key, config_value FROM SystemConfig;"
                ).fetchall()
                for row in rows:
                    result[row["config_key"]] = row["config_value"]
        except Exception:
            # If the table doesn't exist yet (very fresh DB before db_init
            # has been run), we proceed with all defaults. The boot
            # sequence in main.py runs db_init first anyway, so this branch
            # is mostly defensive.
            logger.exception(
                "Failed to read SystemConfig — using all defaults."
            )
        return result

    @staticmethod
    def _coerce_int(raw: Any, default: int, key_name: str) -> int:
        """Parse a SystemConfig value as int; fall back to default on failure."""
        if raw is None:
            return default
        try:
            return int(raw)
        except (TypeError, ValueError):
            logger.warning(
                "SystemConfig key %r has non-integer value %r — using default %d.",
                key_name, raw, default,
            )
            return default
        