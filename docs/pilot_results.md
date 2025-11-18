# Pilot Results â€” Short Update

**Scope:** Baselines + first pilot model on 3â€“5 stations.  
**Horizons:** 1, 3, 6, 24 hours.

## Baselines
- See `reports\persistence_macro.csv` (macro MAE/RMSE by horizon).
- Brief comment on which horizons look hardest/easiest.

## Threshold Events
- See `reports\threshold_event_counts.csv` (â‰¥3 g/L; â‰¥4 g/L per station).
- Stations viable for classification vs regression-only.

## Pilot Model (LSTM/TCN)
- Data split: chronological (last 15% test).
- Features: lags (1â€“24 h), rolling mean/std (3â€“24 h), time-of-day / day-of-year.
- Results: include 2â€“3 figures from `figures\`.
- Comparison vs persistence at 24 h: X stations beat baseline by Y%.

## Notes & Next Steps
- What to improve (features, tuning, data QC)
- Risks / blockers (if any)

