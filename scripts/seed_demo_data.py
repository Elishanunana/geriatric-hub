"""
seed_demo_data.py
=================
Demo data seeder. Inserts a realistic ElderProfile and a MedicationSchedule
whose `time_due` is calculated dynamically to fall ~45 seconds after this
script is run. This guarantees the ReminderScheduler will fire a live TTS
prompt during the academic defense demo.

Workflow for the panel
----------------------
Terminal A:
    python scripts/seed_demo_data.py
    python -m src.main

Terminal B (optional):
    python -m src.dev_console

Within ~45 seconds of starting `main.py`, the reminder scheduler will
detect the medication is due, fire the Twi prompt over the speaker, and
wait for a confirmation. The operator can then either:
  • Type `1` in Terminal A's [MIC] prompt to confirm the dose, or
  • Use Terminal B option [4] to inject DOSE_CONFIRMED, or
  • Do nothing and watch the timeout / retry logic engage.

Author: Wise (Asumang Pobi Godwin) — KNUST COE 497
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta

from src.data_management.db_init       import initialize_database
from src.data_management.db_connection import get_connection
from src.data_management.repositories  import (
    ElderProfileRepo, ElderProfile,
    MedicationScheduleRepo, MedicationSchedule,
)


# ===========================================================================
# Configuration
# ===========================================================================

DEMO_ELDER_NAME       = "Maame Akua Owusu"
DEMO_LANGUAGE         = "twi"
DEMO_CAREGIVER_PHONES = "+233244111111,+233244222222"

DEMO_DRUG_NAME = "Amlodipine"
DEMO_DOSAGE    = "5mg"

# How many seconds after this script runs the medication should be due.
SECONDS_UNTIL_DUE = 45


# ===========================================================================
# Helpers
# ===========================================================================

def upsert_demo_elder() -> int:
    """
    Ensure exactly one demo elder exists and has the demo caregiver phones.
    Returns the elder_id.
    """
    repo = ElderProfileRepo()
    existing = repo.fetch_first()

    if existing is None:
        elder_id = repo.insert(ElderProfile(
            name             = DEMO_ELDER_NAME,
            language         = DEMO_LANGUAGE,
            caregiver_phones = DEMO_CAREGIVER_PHONES,
        ))
        print(f"  ✓ Inserted ElderProfile  (elder_id={elder_id}, name='{DEMO_ELDER_NAME}')")
        return elder_id

    # Update name and caregiver phones to the demo values, leaving other fields
    # alone. We do this directly with a parameterised UPDATE rather than adding
    # a one-off method to the repo.
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE ElderProfile
               SET name             = ?,
                   language         = ?,
                   caregiver_phones = ?,
                   last_modified    = strftime('%Y-%m-%dT%H:%M:%fZ','now')
             WHERE elder_id = ?;
            """,
            (DEMO_ELDER_NAME, DEMO_LANGUAGE, DEMO_CAREGIVER_PHONES, existing.elder_id),
        )
    print(
        f"  ✓ Updated existing ElderProfile  (elder_id={existing.elder_id}, "
        f"name='{DEMO_ELDER_NAME}')"
    )
    return existing.elder_id


def schedule_demo_medication(elder_id: int) -> int:
    """
    Insert (or update) a medication scheduled to fire SECONDS_UNTIL_DUE
    after now. Returns the schedule_id.

    The reminder scheduler queries fetch_due_within(60) every poll
    interval, so any time_due falling within the next minute (and on
    today's day-of-week, which 'DAILY' always does) will trigger.
    """
    now    = datetime.now()
    due_at = now + timedelta(seconds=SECONDS_UNTIL_DUE)
    time_due_hhmm = due_at.strftime("%H:%M")

    repo = MedicationScheduleRepo()
    schedule_id = repo.upsert_from_payload(
        MedicationSchedule(
            elder_id     = elder_id,
            drug_name    = DEMO_DRUG_NAME,
            dosage       = DEMO_DOSAGE,
            time_due     = time_due_hhmm,
            days_of_week = "DAILY",
            active       = 1,
        ),
        sync_method   = "hub_local",
        prescribed_by = "pharmacist",
    )
    print(
        f"  ✓ Scheduled  {DEMO_DRUG_NAME} {DEMO_DOSAGE}  "
        f"@ {time_due_hhmm}  (schedule_id={schedule_id})"
    )
    print(f"     Now:        {now.strftime('%H:%M:%S')}")
    print(f"     Due at:     {due_at.strftime('%H:%M:%S')}  "
          f"(in ~{SECONDS_UNTIL_DUE} seconds)")
    return schedule_id


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    print()
    print("═" * 70)
    print("  DEMO DATA SEEDER — Geriatric Care Hub")
    print("═" * 70)
    print()

    # Make sure the schema exists. Idempotent — see Task 1.
    initialize_database(verbose=False)

    elder_id = upsert_demo_elder()
    print()
    schedule_demo_medication(elder_id)

    print()
    print("─" * 70)
    print("  Now run:  python -m src.main")
    print(f"  The reminder will fire approximately {SECONDS_UNTIL_DUE} seconds")
    print("  after the hub completes its boot sequence.")
    print("─" * 70)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
    