# ChinaMacro2026: Quant-Driven Verification Pipeline

### Project Overview
A systematic macro-quant engine designed to verify and challenge sell-side growth narratives (specifically GS 2026 Outlook). The pipeline utilizes state-dependent lead-lag models and Monte Carlo simulations to quantify "Transmission Decay" in China's credit and fiscal impulses.

### Key Features
* **Variant Perception Engine:** Dissects the 28bps gap between consensus GDP forecasts (4.8%) and our baseline (4.52%).
* **Property Stabilization Matrix:** A city-level heatmap analysis of 70+ cities tracking dispersion and tail-risk shares in residential pricing.
* **Transmission Multiplier Analysis:** Quantifies the decaying correlation between TSF (Total Social Financing) and Industrial Value Added (IVA).
* **Automated Reporting:** End-to-end pipeline in Python (`run.py`) generating institutional-grade PDF analysis from raw Excel/CSV inputs.

### Tech Stack
* **Core:** Python (Pandas, NumPy, Scipy, Matplotlib/Seaborn)
* **Architecture:** Config-driven pipeline (`config.yaml`) for modular task execution (Property, External, Domestic, Nowcast).
* **Data Source:** Proprietary dataset curated from NBS (National Bureau of Statistics), PBoC, and GAC (General Administration of Customs).

### Pipeline Execution
```bash
python -m src.run --config config.yaml
