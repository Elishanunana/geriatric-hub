"""
sms_transport.py
================
Outbound SMS dispatcher — the Hub→App side of the SMS synchronisation
pathway described in Section 3.5.4 of the project report. Complements
the inbound `SMSPayloadHandler` we built in Task 7.

Responsibilities
----------------
1. Take a SyncQueueEntry destined for SMS transport.
2. Build the canonical pipe-delimited payload, base64-encode the JSON
   payload field to safely contain arbitrary characters (including pipes
   and newlines), and append an HMAC-SHA256 tag computed with the same
   shared key the inbound handler verifies against.
3. Resolve the destination phone numbers from ElderProfileRepo — every
   registered caregiver phone receives the message (the multi-caregiver
   model documented in the report).
4. Dispatch via the GSM module and report per-recipient success/failure.
5. Log the outbound event for the audit trail.

Format (mirrors the Task 7 inbound contract)
--------------------------------------------
    SYNC|<entity_type>|<change_type>|<change_id>|<timestamp>|<payload_b64>|HMAC=<hex>

The base64 wrapping is deliberate: it lets the payload carry any JSON
without pipe-character collisions breaking the parser. The trade-off is
size — most outbound messages will exceed the 160-character single-segment
SMS limit. The report acknowledges this in Section 3.5.4 ("Where the
payload exceeds this limit, a defined multi-part segmentation scheme...
is used"). For the mock phase we log a warning when payloads exceed the
limit and proceed regardless; multi-part segmentation is a future
extension when integrating with a real SIM800L.

Author: Wise (Asumang Pobi Godwin) — KNUST COE 497
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from src.data_management.repositories import (
    ElderProfileRepo,
    EventLogRepo,
    SyncQueueRepo,
    SyncQueueEntry,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Configuration
# ===========================================================================

PAYLOAD_SENTINEL          = "SYNC"
HMAC_PREFIX               = "HMAC="
DEV_MOCK_HMAC_TAG_VALUE   = "MOCK"
DEFAULT_MAX_PAYLOAD_CHARS = 160     # Single-segment SMS limit (informational).


# ===========================================================================
# Result types
# ===========================================================================

@dataclass
class RecipientOutcome:
    phone:    str
    success:  bool


@dataclass
class DispatchResult:
    change_id:     str
    payload_chars: int
    recipients:    List[RecipientOutcome] = field(default_factory=list)
    overall_ok:    bool = False
    error:         Optional[str] = None


# ===========================================================================
# SMSTransport
# ===========================================================================

class SMSTransport:
    """
    Stateless wrapper around the GSM module for outbound sync dispatch.
    All dependencies are injected so production and test contexts share
    the exact same code path.
    """

    def __init__(
        self,
        gsm: Any,                                     # MockGSMModule / GSMModule
        elder_repo: Optional[ElderProfileRepo] = None,
        event_repo: Optional[EventLogRepo]     = None,
        sync_repo:  Optional[SyncQueueRepo]    = None,
        *,
        hmac_key: str = "",
        dev_mock_hmac: bool = False,
        max_payload_chars: int = DEFAULT_MAX_PAYLOAD_CHARS,
    ):
        """
        Parameters
        ----------
        gsm : injected GSM adapter exposing .send_sms(recipient, body).
        *_repo : repository instances; default-constructed if omitted.
        hmac_key : the shared key derived from the pairing token (Section 3.5.5).
        dev_mock_hmac : if True, the literal tag 'MOCK' is appended in place
                        of a real HMAC. MUST be False in production.
        max_payload_chars : informational SMS size limit. Exceeding it logs
                            a warning but does not block dispatch.
        """
        self._gsm        = gsm
        self._elder_repo = elder_repo or ElderProfileRepo()
        self._event_repo = event_repo or EventLogRepo()
        self._sync_repo  = sync_repo  or SyncQueueRepo()

        self._hmac_key          = hmac_key.encode("utf-8") if hmac_key else b""
        self._dev_mock_hmac     = dev_mock_hmac
        self._max_payload_chars = max_payload_chars

        if not dev_mock_hmac and not self._hmac_key:
            raise ValueError(
                "SMSTransport requires either hmac_key or dev_mock_hmac=True."
            )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def dispatch(
        self,
        entry: SyncQueueEntry,
        *,
        elder_id: Optional[int] = None,
    ) -> DispatchResult:
        """
        Build, sign, and send the SMS for one SyncQueue entry. Sends to
        every registered caregiver phone for the elder. Per-recipient
        outcomes are returned so the caller (typically SyncEngine) can
        decide whether to mark_synced or mark_failed.

        Convention: 'overall_ok' is True iff at least ONE recipient
        succeeded. The report's Hub→App use case is "the caregiver app
        receives the event"; if any caregiver receives it, the event has
        been propagated. Recipients that failed get retried on the next
        sync window via the standard mark_failed → pending recovery loop.
        """
        result = DispatchResult(change_id=entry.change_id, payload_chars=0)

        # 1. Build & sign the payload.
        try:
            body = self._build_signed_body(entry)
        except Exception as exc:
            logger.exception(
                "Failed to build signed SMS body for change_id=%s",
                entry.change_id,
            )
            result.error = f"build_failed: {exc}"
            return result

        result.payload_chars = len(body)
        if len(body) > self._max_payload_chars:
            logger.warning(
                "Outbound SMS payload is %d chars (>%d single-segment limit). "
                "Multi-part segmentation not implemented in mock phase — sending whole.",
                len(body), self._max_payload_chars,
            )

        # 2. Resolve recipients.
        recipients = self._resolve_recipients(elder_id)
        if not recipients:
            logger.error(
                "No caregiver phones registered — outbound SMS for change_id=%s "
                "cannot be dispatched.", entry.change_id,
            )
            result.error = "no_recipients"
            self._event_repo.insert(
                EventLogRepo.SYSTEM_FAULT,
                details={
                    "subsystem": "SMSTransport",
                    "stage":     "resolve_recipients",
                    "change_id": entry.change_id,
                    "reason":    "no_caregiver_numbers",
                },
            )
            return result

        # 3. Dispatch to each recipient.
        for phone in recipients:
            ok = False
            try:
                ok = bool(self._gsm.send_sms(phone, body))
            except Exception:
                logger.exception("send_sms raised for recipient %s", phone)
                ok = False
            result.recipients.append(RecipientOutcome(phone=phone, success=ok))

        result.overall_ok = any(r.success for r in result.recipients)

        # 4. Log the dispatch outcome to the audit trail.
        self._event_repo.insert(
            "sms_outbound_dispatched" if result.overall_ok
            else "sms_outbound_failed",
            details={
                "change_id":       entry.change_id,
                "entity_type":     entry.entity_type,
                "change_type":     entry.change_type,
                "payload_chars":   result.payload_chars,
                "recipients_total": len(result.recipients),
                "recipients_ok":    sum(1 for r in result.recipients if r.success),
                "recipients_fail":  sum(1 for r in result.recipients if not r.success),
                "succeeded":        [r.phone for r in result.recipients if r.success],
                "failed":           [r.phone for r in result.recipients if not r.success],
            },
        )
        return result

    # -----------------------------------------------------------------------
    # Internal — payload construction
    # -----------------------------------------------------------------------

    def _build_signed_body(self, entry: SyncQueueEntry) -> str:
        """
        Build:
            SYNC|<entity_type>|<change_type>|<change_id>|<timestamp>|<payload_b64>|HMAC=<hex>

        The payload is base64-encoded JSON of whatever the entry carries
        (typically a dict) so pipe characters and embedded JSON don't
        break the parser on the receiving side.
        """
        timestamp = entry.timestamp or _utcnow_iso()

        # Normalise the payload: if it's already a JSON string, parse and
        # re-emit canonical compact JSON; if it's blank, use {}.
        if entry.payload:
            try:
                payload_obj = json.loads(entry.payload)
            except (json.JSONDecodeError, TypeError):
                # Not valid JSON — wrap it as a string.
                payload_obj = {"raw": entry.payload}
        else:
            payload_obj = {}

        compact_json = json.dumps(payload_obj, ensure_ascii=False,
                                  separators=(",", ":"), default=str)
        payload_b64 = base64.b64encode(
            compact_json.encode("utf-8")
        ).decode("ascii")

        canonical = "|".join([
            PAYLOAD_SENTINEL,
            entry.entity_type,
            entry.change_type,
            entry.change_id,
            timestamp,
            payload_b64,
        ])

        if self._dev_mock_hmac:
            tag = DEV_MOCK_HMAC_TAG_VALUE
        else:
            tag = hmac.new(
                self._hmac_key,
                canonical.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()

        return canonical + "|" + HMAC_PREFIX + tag

    # -----------------------------------------------------------------------
    # Internal — recipient resolution
    # -----------------------------------------------------------------------

    def _resolve_recipients(self, elder_id: Optional[int]) -> List[str]:
        """
        Resolve the list of caregiver phone numbers to send to. If
        elder_id is None, falls back to the first elder — appropriate for
        single-elder deployments which the report assumes are the typical
        case.
        """
        if elder_id is None:
            elder = self._elder_repo.fetch_first()
        else:
            elder = self._elder_repo.fetch_by_id(elder_id)
        if elder is None:
            return []
        return self._elder_repo.caregiver_phones(elder.elder_id)


# ===========================================================================
# Module helpers
# ===========================================================================

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )
