# Mekong Salinity â€” Repo Quick Start

## Environment (Windows)
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run baselines (outputs to `reports\`)

```bat
python scripts\persistence_baseline.py --root "C:\Users\User\Desktop\Disertation" --horizons 1,3,6,24
python scripts\count_threshold_events.py --root "C:\Users\User\Desktop\Disertation"
```

## Weekly docs to keep updated

* `docs\STATUS.md` â€” rolling log (queries, decisions, blockers, next actions)
* `docs\pilot_results.md` â€” 1â€“2 page update for supervisor (export to PDF for email)

## Folders

* `DATA\`           private dataset (do not share)
* `scripts\`        analysis scripts (no notebooks required)
* `reports\`        CSV outputs and short markdown summaries
* `figures\`        PNG charts from pilot models / diagnostics
* `viewer\`         the HTML viewer app (local only)

**Privacy:** Data is private use only. Do not publish raw data or repo.

