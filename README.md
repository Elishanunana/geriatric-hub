# Resilient, Offline-First Assistive Ecosystem for Geriatric Care

> An undergraduate final-year engineering project — Department of Computer Engineering, Kwame Nkrumah University of Science and Technology (KNUST), College of Engineering. Submitted in partial fulfilment of the requirements for the BSc. Computer Engineering degree, COE 497.

---

## 1. Overview

The Resilient, Offline-First Assistive Ecosystem is a hub-and-spoke geriatric care system designed for elderly Ghanaians with Mild Cognitive Impairment (MCI) in Kumasi, Ghana, and similar resource-constrained environments across sub-Saharan Africa.

The system addresses three structural barriers that have prevented the adoption of existing assistive technologies among elderly Ghanaians:

- **Infrastructure mismatch** — most existing solutions require continuous internet connectivity and stable mains power, neither of which is reliably available in the target deployment context.
- **Usability and literacy mismatch** — existing solutions assume touchscreen smartphones, English-language interfaces, and digital literacy that the target population does not have.
- **Clinical-cognitive load** — wearable devices and app-based care routines exceed the cognitive capacity of MCI patients, paradoxically excluding the population with the greatest clinical need.

The system comprises a **stationary voice hub** running on a Raspberry Pi 4 single-board computer (currently simulated via hardware mocks), and a **caregiver companion mobile application** (specified but out of scope of this repository). The hub interacts with the elderly user through a constrained Asante Twi voice interface, delivers proactive medication reminders and emergency SOS alerts, and synchronises with the caregiver application via either a local Wi-Fi REST channel or a fallback SMS-over-GSM channel — whichever is available at any given moment.

This repository contains the complete software stack of the hub, including the integrated system orchestrator, the formal correctness test suite, and the developer tooling required to demonstrate the system end-to-end.

---

## 2. Architecture

The codebase implements the four-layer architecture defined in Section 3.5 of the project report. Every module maps to exactly one layer, and every cross-layer interaction follows the dependency direction documented below.

```text
┌─────────────────────────────────────────────────────────────────────┐
│                       Voice Engine Layer                            │
│              Vosk keyword spotter + Piper Twi TTS                   │
│                   (currently in Mock Phase)                         │
│                                                                     │
│         src/hardware_mocks/{mock_microphone, mock_speaker}.py       │
└─────────────────────────────────────────────────────────────────────┘
↑
┌─────────────────────────────────────────────────────────────────────┐
│                     Control Logic Layer                             │
│                                                                     │
│   • ReminderScheduler       — cron-style medication prompt loop     │
│   • SOSHandler              — high-priority emergency interrupt     │
│   • CommandDispatcher       — routes voice commands to actions      │
│   • SMSPayloadHandler       — inbound SMS validation pipeline       │
│                                                                     │
│                    src/control_logic/*.py                           │
└─────────────────────────────────────────────────────────────────────┘
↑
┌─────────────────────────────────────────────────────────────────────┐
│                    Communication Layer                              │
│                                                                     │
│   • LocalRestAPI            — Flask server on the AP interface      │
│   • SMSTransport            — outbound SMS dispatcher               │
│   • ConnectivityArbiter     — pathway selection + urgency promotion │
│   • SyncEngine              — drives outbound SyncQueue dispatch    │
│                                                                     │
│                    src/communication/*.py                           │
└─────────────────────────────────────────────────────────────────────┘
↑
┌─────────────────────────────────────────────────────────────────────┐
│                    Data Management Layer                            │
│                                                                     │
│   • db_init.py              — schema, indexes, append-only triggers │
│   • db_connection.py        — connection factory with WAL + FK ON   │
│   • repositories.py         — CRUD for the five canonical tables    │
│                                                                     │
│   Tables: ElderProfile, MedicationSchedule, EventLog (append-only), │
│           SyncQueue, SystemConfig, DevCommandQueue (debug IPC)      │
│                                                                     │
│                    src/data_management/*.py                         │
└─────────────────────────────────────────────────────────────────────┘
```

**Persistence** is provided by a single SQLite database (`data/geriatric_hub.db`) operating in WAL mode for concurrent reader/writer access. The `EventLog` table is enforced as append-only at the database level via SQLite triggers — records can be inserted but not updated or deleted (with one narrow exception: advancing `synced_flag` from 0 to 1 for the sync subsystem).

**Voice Engine** integration with Vosk and Piper is currently in the Mock Phase. The mock implementations in `src/hardware_mocks/` expose the same public method signatures the real drivers will expose, so the swap to Raspberry Pi hardware is a single-line import change in the orchestrator.

**The Communication Layer's hybrid sync protocol** — Section 3.5.4 of the report — chooses between Wi-Fi REST (when the caregiver app is associated with the hub's Access Point) and SMS-over-GSM (when the caregiver is geographically remote), with an urgency-aware fallback that promotes SOS and missed-dose events from Wi-Fi to SMS if the urgency budget is exceeded while Wi-Fi is unavailable.

**The system orchestrator** (`src/main.py`) constructs every subsystem in dependency order, installs a fan-out dispatcher so the microphone's events reach the ReminderScheduler, SOSHandler, and CommandDispatcher simultaneously with exception isolation, and provides graceful shutdown on SIGINT / SIGTERM.

---

## 3. Quick Start

### Prerequisites

- Python 3.11 or newer (Python 3.13 verified)
- A Unix-like shell or Windows PowerShell

### Setup

Clone the repository and create an isolated virtual environment:

```bash
git clone <repository-url> geriatric-hub
cd geriatric-hub
python -m venv venv
```

Activate the virtual environment:

```bash
# Linux / macOS
source venv/bin/activate

# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Initialise the Database
Apply the SQLite schema to a fresh `data/geriatric_hub.db` file. This step is idempotent — running it on an existing database preserves all data:

```bash
python -m src.data_management.db_init
```
You will see the five tables and the append-only triggers being installed, followed by a self-test that confirms INSERT succeeds, UPDATE of an immutable column is correctly blocked, and DELETE is correctly blocked.

### Run the Hub
Start the integrated system orchestrator:

```bash
python -m src.main
```
The hub will walk through its eight-step boot sequence (configuration, repositories, hardware adapters, communication layer, control logic, cross-wiring, background threads, run loop). Once the `HUB ONLINE` banner appears, the system is fully operational and accepting voice commands at the `[MIC] >` prompt.

Press `Ctrl+C` to trigger a graceful shutdown.

---

## 4. Panel Defense / Demo Instructions

The most compelling demonstration uses a three-terminal layout that lets you drive the system end-to-end without touching the microphone prompt directly. This keeps the hub's terminal output clean and chronological for the panel to follow.

**Step 1 — Seed Demo Data (Terminal 0)**
Insert a realistic elder profile and a medication schedule due ~45 seconds in the future. This guarantees the ReminderScheduler will fire a live TTS prompt during the demo:

```bash
python -m scripts.seed_demo_data
```
Output:
```text
✓ Updated existing ElderProfile (elder_id=1, name='Maame Akua Owusu')
✓ Scheduled  Amlodipine 5mg  @ 14:50  (schedule_id=5)
   Now:        14:49:45
   Due at:     14:50:30  (in ~45 seconds)
```

**Step 2 — Start the Hub (Terminal A)**
```bash
python -m src.main
```
The eight-step boot banner walks through every subsystem coming online, ending with `HUB ONLINE`. Within 30–45 seconds, the seeded reminder will fire automatically, producing a live Twi TTS prompt over the speaker mock.

**Step 3 — Open the Developer Console (Terminal B)**
The dev console runs in a separate process and communicates with the hub via a SQLite-based IPC table (`DevCommandQueue`). It lets the operator inject events without typing at the hub's `[MIC] >` prompt:

```bash
python -m src.dev_console
```
The console offers a numbered menu that triggers the following events against the running hub:

| Console Option | Effect on the Hub |
|---|---|
| `[1]`, `[2]` | Toggle simulated Wi-Fi caregiver-connection state |
| `[3]` | Inject voice command `Boa me!` → SOS pathway fires |
| `[4]` | Inject voice command `Yε, mafa m'aduru` → confirms a pending dose |
| `[5]` | Inject voice command `Mfaa m'aduru nkaa` → declines a pending dose |
| `[6]`, `[7]` | Inject appliance ON / OFF voice commands |
| `[8]` | Inject schedule readback voice command |
| `[9]` | Inject repeat-last voice command |
| `[10]` | Trigger physical SOS button (GPIO interrupt simulation) |
| `[11]` | Inject a valid pharmacist-signed inbound SMS payload |
| `[12]` | Inject SMS from an unknown number — should be rejected |
| `[13]` | Inject malformed SMS payload — should be rejected at schema stage |

### Recommended Demo Sequence (5 minutes)
1. Show Terminal A's eight-step boot log to introduce the architecture.
2. From Terminal B, press `[1]` to connect the caregiver app over Wi-Fi.
3. From Terminal B, press `[11]` to inject a pharmacist-entered medication via SMS — Terminal A shows the full validation pipeline (origin check, HMAC, idempotency, atomic apply, SIM cleanup).
4. Wait for the seeded reminder to fire. Terminal A speaks the Amlodipine prompt.
5. From Terminal B, press `[4]` to confirm the dose. Terminal A logs `dose_confirmed` and queues a Hub→App sync record.
6. From Terminal B, press `[2]` to disconnect Wi-Fi.
7. From Terminal B, press `[3]` to trigger an SOS — Terminal A flashes the LED red, dispatches SMS to caregivers, and speaks the Twi reassurance.
8. Wait ~60 seconds. The SyncEngine will autonomously detect that the SOS event has exceeded its urgency budget while Wi-Fi remains down, promote the entry to SMS, and dispatch it. The panel watches the system make a transport decision in real time.
9. From Terminal B, press `[1]` to reconnect Wi-Fi.
10. Press `Ctrl+C` in Terminal A to trigger graceful shutdown — every subsystem stops in reverse-dependency order.

### Optional Verification (Browser or Terminal C)
While the hub is running, the local REST API is reachable at `http://localhost:5000`. The `/api/v1/health` endpoint is unauthenticated and useful for proving the Wi-Fi sync endpoint is live:

```bash
curl http://localhost:5000/api/v1/health
```
Expected response:
```json
{
  "status": "ok",
  "system": "geriatric_hub",
  "api_version": "v1",
  "server_time": "2026-04-29T..."
}
```

---

## 5. Formal Verification

Section 3.7.3 of the project report commits the system to a synchronisation correctness target of ≥ 98% across ten edge-case scenarios, each executed three times — a total of thirty test executions, of which at most one failure is permitted.

The test suite at `tests/test_integration.py` implements all ten scenarios as pytest cases parameterised over three runs each. The scenarios cover both transport pathways and the boundary conditions specified in the report:

| # | Scenario | Pathway |
|---|---|---|
| 1 | Valid SMS payload — registered caregiver, valid HMAC | SMS |
| 2 | Duplicate change_id — idempotent rejection | SMS |
| 3 | Unknown sender — origin verification rejection | SMS |
| 4 | Tampered HMAC — cryptographic verification rejection | SMS |
| 5 | Wi-Fi REST schedule sync — multi-record batch | Wi-Fi REST |
| 6 | REST API batch limit enforcement | Wi-Fi REST |
| 7 | SOS acknowledgement clearing the LED | Wi-Fi REST |
| 8 | Arbiter promotion — urgent overdue SOS (Wi-Fi → SMS) | Hybrid |
| 9 | Arbiter restraint — non-urgent entries not promoted | Hybrid |
| 10 | EventLog append-only invariant under raw SQL bypass | Persistence |

### Running the Suite
```bash
pytest tests/test_integration.py -v
```

### Current Status
**30 passed in 2.74s** All thirty executions across all ten scenarios pass. This corresponds to 100% correctness, which exceeds the report's ≥ 98% commitment by the maximum possible margin.

The suite uses pytest's `monkeypatch` and `tmp_path` fixtures to give each test a fresh, isolated SQLite database, so there is no state leakage between tests and results are fully reproducible. Tests drive the production subsystems directly — `LocalRestAPI.test_client()` for HTTP routes and `SMSPayloadHandler.process_once()` for the inbound SMS pipeline — so no background threads are spawned during testing and outcomes are deterministic.

To produce a permanent record of a test run, redirect output to a file:
```bash
pytest tests/test_integration.py -v > tests/section_3_7_3_results.txt 2>&1
```

---

## 6. Voice Commands

The constrained Asante Twi command vocabulary (Table 3.2 of the project report) was developed to deliberately avoid open-domain ASR, which is unreliable for elderly speakers and for low-resource languages. Each command is phonetically distinct from every other and maps to exactly one system action.

| Action | Twi Phrase | Approximate English Meaning | System Action |
|---|---|---|---|
| `ACTION_DOSE_CONFIRMED` | `Yε, mafa m'aduru` | Yes, I have taken my medicine | Log dose confirmation |
| `ACTION_DOSE_MISSED` | `Mfaa m'aduru nkaa` | I have not yet taken it | Log missed dose; reschedule alert |
| `ACTION_SOS` | `Boa me!` | Help me! | Trigger SOS — SMS dispatch + flashing red LED |
| `ACTION_APPLIANCE_ON` | `Sua fitaa no` | Switch on the light | Activate relay (appliance on) |
| `ACTION_APPLIANCE_OFF` | `Sua fitaa no na` | Switch off the light | Deactivate relay (appliance off) |
| `ACTION_READ_SCHEDULE` | `Aduru bεn na mefa?` | What medicine do I take? | Read today's medication schedule aloud |
| `ACTION_REPEAT_LAST` | `Mesrε wo, yε san bio` | Please repeat that | Re-play the last spoken prompt |

The final phrasing of these commands is subject to validation in the User Needs Assessment phase (Section 3.2 of the report), conducted with elderly Ghanaian participants in Kumasi. The command set above represents the preliminary vocabulary; phonetic distinctiveness criteria are enforced in line with Section 2.6 of the report ("avoiding pairs of commands that differ primarily in tone rather than segmental content").

---

### Project Information
**Authors:**
- Danso Nicole Kusiwaa — 1821022
- Akabua Elisha Nunana — 1815222
- Asumang Pobi Godwin — 1818822

**Supervisor:** Dr. Theresa S. A. Adjaidoo  
**Institution:** Department of Computer Engineering, Faculty of Electrical and Computer Engineering, College of Engineering, Kwame Nkrumah University of Science and Technology (KNUST), Kumasi, Ghana.  
**Date:** February 2026
```