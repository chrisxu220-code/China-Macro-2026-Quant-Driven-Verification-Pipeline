# Feature Store (Task C)

This folder standardizes the two core constructs used in the deck / note:

1) Fiscal Fuse
- Idea: policy/fiscal impulse leads the real economy by ~12 months.
- Implementation: `src/features/fiscal.py`
  - `build_fiscal_fuse(wide, spec)`
  - `lead_lag_corr(wide, x, y, max_lag=18)`

2) Paradigm Shift Index
- Idea: "new engines" vs "old economy" capex intensity composite.
- Implementation: `src/features/structural_shift.py`
  - robust scaling (median/MAD) by default
  - composite = diff(new_comp - old_comp) by default

3) Seasonality / CNY Distortion Handling
- Implementation: `src/features/seasonality.py`
  - default method merges Jan+Feb into Feb-anchored value:
    - flows: sum
    - rates/indices: mean

## Expected Inputs
- long-form timeseries exported by Task B:
  - columns: dataset, series, date, value (optional: unit, frequency, notes)

## Recommended Pipeline Shape (next step)
- Load `output/processed/timeseries_long.csv`
- Apply seasonality adjustment (merge Jan-Feb)
- Pivot to wide monthly matrix
- Compute:
  - fiscal fuse dataframe (tsf lead 12m vs target)
  - paradigm shift index dataframe
- Save to `output/features/*.csv` and optionally plots