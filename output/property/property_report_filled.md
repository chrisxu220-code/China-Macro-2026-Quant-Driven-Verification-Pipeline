# China Property: Stabilization Signals Improving, but Breadth Remains Narrow  
*(Quant Verification Engine — Property Module)*

**As of:** 2026-01-26  
**Data:** City-level YoY price indices (new vs existing) + monthly sales / inventory (if available)  
**Method:** Distribution-based regime signal + consistency checks vs transactions and inventory

---

## Executive Summary (Macro / Trading Take)

- The distribution-based regime engine indicates a transition from broad deterioration to **policy-supported stabilization** in 2025-03–2025-05, with stabilization probability rising to **0.34** (ALL (pooled new+existing)).
- The stabilization remains **narrow and asymmetric**: downside tail pressure is easing and dispersion is compressing, but **breadth (YoY ≥ 100) remains limited**, consistent with a **managed price floor rather than a broad cyclical recovery**.
- Cross-validation versus monthly transactions is **directionally informative but not conclusive**. Flow-through into sales remains uneven, implying that current stabilization is **fragile and dependent on continued policy support** rather than organic demand recovery.

---

## 1) Core Signal — Stabilization Probability (Regime)

![Stabilization Probability](output/property/figures/stabilization_probability_signal.png)

**Latest (2025-12):**
- Stabilization probability: **0.34**
- City coverage (n): **65**
- Construction: probability aggregates **Δ median**, **Δ dispersion**, **Δ breadth**, and **Δ downside tail**, mapped via robust z-scores and logistic scaling.

**Interpretation:**  
The sustained improvement in the regime probability through 2025-03–2025-05 is consistent with a shift from left-tail-driven declines toward a **range-bound stabilization regime**. Persistence at elevated levels into recent months suggests the floor is holding, but does not yet constitute evidence of an early-cycle upturn.

---

## 2) Distribution Anatomy — Tails, Breadth, Dispersion

### 2.1 Tail / Breadth Shares (New Homes)

![Tail shares — new](output/property/figures/tail_shares_new.png)

**Latest (2025-12):**
- Breadth (YoY ≥ 100): **7.7%**
- Downside tail (YoY ≤ 95): **10.8%**
- Neutral-band share (95–100): **81.5%**

**Read-through:**  
The decline in downside tail pressure combined with a rising neutral-band share points to **flooring dynamics**. However, limited breadth implies that stabilization is being achieved primarily through **compression toward the neutral range**, rather than broad-based price appreciation.

### 2.2 Dispersion (MAD) — New Homes

![Dispersion — new](output/property/figures/dispersion_new.png)

**Read-through:**  
Ongoing dispersion compression suggests price declines are becoming less idiosyncratic across cities, consistent with a **policy-coordinated stabilization regime**. This pattern typically characterizes late-downturn or early-stabilization phases, not a sustained expansion.

---

### 2.3 Tail / Breadth Shares (Existing Homes)

![Tail shares — existing](output/property/figures/tail_shares_existing.png)

### 2.4 Dispersion (MAD) — Existing Homes

![Dispersion — existing](output/property/figures/dispersion_existing.png)

**Read-through (Existing Homes):**  
Existing-home prices remain structurally weaker than new homes, with a **large downside tail and very limited breadth**. This divergence highlights continued stress in resale liquidity, household balance sheets, and expectations, and suggests that stabilization has been more effective in the **primary (new-home) market** than in the secondary market.

---

## 3) Cross-Validation — Signal vs Monthly Transactions (Indicative)

![Validation: signal vs sales](output/property/figures/validation_price_vs_sales.png)

**What we look for:**
- Whether rising stabilization probability is accompanied by improving monthly sales activity (levels and/or momentum).
- Persistent divergence would indicate a **policy floor without volume confirmation**, historically associated with fragile stabilization episodes.

**Interpretation:**  
Recent alignment is directionally supportive but remains uneven, underscoring that stabilization has yet to translate into a durable recovery in transaction flows.

---

## 4) Lead/Lag Evidence — Diagnostic Only

![Lead/Lag Correlation](output/property/figures/validation_leadlag_corr.png)

**Summary (indicative, not causal):**
- Best lead (months): **3**
- Peak correlation: **-0.13**

**Interpretation:**  
Given short samples and definitional noise in monthly sales data, lead/lag results should be treated as **diagnostic rather than predictive**. At this stage, evidence for systematic leading behavior remains **mixed**.

---

## 5) Inventory Check — Durability Test (If Available)

![Inventory Destocking](output/property/figures/inventory_destocking_check.png)

**Read-through:**  
Sustained Δinventory < 0 would confirm that stabilization is being reinforced by **absorption and destocking**, rather than maintained purely through pricing or administrative measures. Inventory series should be interpreted with care due to level-versus-change reporting issues.

---

## Appendix (Optional but High-Value)

### A1) Event Study — Bottoming / Turning Point (New Homes)

![Event study](output/property/figures/event_study_bottoming_newhome.png)

### A2) Mechanism Dashboard — Dispersion vs Breadth/Tail Proxies

![Proxy linkage](output/property/figures/proxy_linkage_dashboard.png)

### A3) Cross-Section — Who Is Recovering? (Heatmap)

![YoY recovery heatmap](output/property/figures/yoy_recovery_dashboard_heatmap_with_delta.png)

**How to read:**  
Cities are ranked by ΔYoY (last–first month). The chart helps distinguish whether stabilization is **broad-based** or concentrated in a subset of markets, and whether improvements reflect genuine recovery or convergence toward a managed floor.

---

*Source: internal quant verification engine. For discussion only.*
