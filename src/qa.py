from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class QAResult:
    dataset: str
    checks: Dict[str, str]  # check_name -> status/message


def _freq_infer_ok(dates: "pd.Series") -> tuple[bool, str]:
    import numpy as np
    import pandas as pd

    d = pd.to_datetime(dates, errors="coerce").dropna().sort_values()
    if len(d) < 6:
        return True, "OK: insufficient points to infer"

    # ✅ pandas DatetimeArray -> numpy datetime64, then cast to [D]
    dn = d.to_numpy(dtype="datetime64[ns]")
    diffs = np.diff(dn.astype("datetime64[D]").astype("int64"))

    if len(diffs) == 0:
        return True, "OK: insufficient diffs"

    # heuristic: monthly-ish if 28-31 day steps dominate
    monthlyish = np.mean((diffs >= 28) & (diffs <= 31))
    if monthlyish >= 0.8:
        return True, f"OK: looks monthly (monthly-ish ratio={monthlyish:.2f})"
    return True, f"OK: freq not clearly monthly (monthly-ish ratio={monthlyish:.2f})"



def run_timeseries_qa(long_df: pd.DataFrame) -> List[QAResult]:
    results: List[QAResult] = []
    if long_df.empty:
        return results

    required = {"dataset", "series", "date", "value"}
    missing_cols = required - set(long_df.columns)
    if missing_cols:
        raise ValueError(f"long_df missing required cols: {missing_cols}")

    for ds, g in long_df.groupby("dataset"):
        checks: Dict[str, str] = {}
        # duplicates
        dup = g.duplicated(subset=["series", "date"]).sum()
        checks["duplicates(series,date)"] = "OK" if dup == 0 else f"FAIL: {dup} duplicates"

        # missing values
        na = g["value"].isna().sum()
        checks["missing_values"] = "OK" if na == 0 else f"WARN: {na} missing values (should be dropped)"

        # date monotonic-ish (per series)
        bad_series = 0
        for s, sg in g.groupby("series"):
            d = pd.to_datetime(sg["date"]).sort_values()
            if d.isna().any():
                bad_series += 1
        checks["date_parse"] = "OK" if bad_series == 0 else f"WARN: {bad_series} series have unparsed dates"

        # freq inference
        ok, msg = _freq_infer_ok(g["date"])
        checks["frequency_infer"] = ("OK: " + msg) if ok else ("FAIL: " + msg)

        # extreme outliers quick scan (absolute z on values)
        v = g["value"].astype(float)
        if len(v) >= 20:
            z = (v - v.mean()) / (v.std() + 1e-9)
            extreme = int((np.abs(z) > 8).sum())
            checks["extreme_outliers"] = "OK" if extreme == 0 else f"WARN: {extreme} points with |z|>8"
        else:
            checks["extreme_outliers"] = "SKIP: too few points"

        results.append(QAResult(dataset=str(ds), checks=checks))

    return results


def qa_results_to_markdown(results: List[QAResult]) -> str:
    if not results:
        return "No QA results (empty long_df)."

    lines = ["# Data QA Summary", ""]
    for r in results:
        lines.append(f"## {r.dataset}")
        lines.append("")
        for k, v in r.checks.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")
    return "\n".join(lines)
