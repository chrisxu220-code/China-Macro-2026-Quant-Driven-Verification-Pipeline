# Model (Task G) — Scenario Accounting Engine

## Why this is not a traditional nowcast
This project’s external-sector dataset is two-point (2024-11 vs 2025-11), which does not support time-series regression or rolling nowcast estimation.  
Instead, we build a **scenario accounting engine** that maps verifiable proxy signals into a growth distribution with explicit parameter assumptions.

## Inputs
- Domestic demand proxy index (Task F)
- Property stabilization probability (Task D; optional)
- Export two-point YoY proxy (Task E)

## Outputs
- Growth distribution + percentiles
- Sensitivity table (factor contributions under baseline assumptions)

## Limitations
- Parameters are **assumptions**, not estimated causal elasticities.
- External sector proxy is two-point and should not be interpreted as persistence.
