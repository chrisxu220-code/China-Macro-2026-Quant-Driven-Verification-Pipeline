from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging
import shutil

# Figures explicitly excluded from final report
EXCLUDED_FIGURES = {
    "external_export_share_slope_both_periods.png",
}

@dataclass(frozen=True)
class ReportSpec:
    repo_root: Path
    report_dir: Path = Path("report")
    report_figures_dir: Path = Path("report/figures")

    # NEW: section-based writing sources (preferred)
    sections_dir: Path = Path("report/sections")
    section_macro: Path = Path("report/sections/macro.md")
    section_property: Path = Path("report/sections/property.md")
    section_external: Path = Path("report/sections/external.md")
    section_domestic: Path = Path("report/sections/domestic.md")
    section_model: Path = Path("report/sections/model.md")

    # Legacy stubs (kept for backward compatibility, but no longer preferred)
    report_property: Path | None = None
    report_external: Path = Path("report/report_external.md")
    report_domestic: Path = Path("report/report_domestic.md")
    report_model: Path = Path("report/report_model.md")

    # Figure source dirs (pipeline outputs)
    macro_fig_dir: Path = Path("output/figures")
    property_fig_dir: Path = Path("output/property/figures")
    external_fig_dir: Path = Path("output/external/figures")
    domestic_fig_dir: Path = Path("output/domestic/figures")
    model_fig_dir: Path = Path("output/model/figures")


def _read_if_exists(path: Path | None) -> str | None:
    if path is None:
        return None
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="ignore")


def _copy_figures(src_dir: Path, dst_dir: Path, prefix: str) -> None:
    """
    Copy *.png from src_dir into dst_dir, prefixing the filename to keep namespaces stable.
    Example: output/property/figures/foo.png -> report/figures/property_foo.png
    """
    if not src_dir.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(src_dir.glob("*.png")):
        dst_name = f"{prefix}_{p.name}"
        if dst_name in EXCLUDED_FIGURES:
            continue
        dst = dst_dir / dst_name
        shutil.copy2(p, dst)


def _property_fallback_section(repo_root: Path) -> str:
    cand = [
        repo_root / "output/property/monthly_regime_table.csv",
        repo_root / "output/property/validation_leadlag_corr.csv",
        repo_root / "output/property/price_distribution_city_panel.csv",
        repo_root / "output/property/price_distribution_metrics.csv",
    ]
    bullets = [f"- `{p.relative_to(repo_root).as_posix()}`" for p in cand if p.exists()]
    if not bullets:
        bullets = ["- (Property artifacts not found; run property_distribution step first)"]

    return "\n".join(
        [
            "## Property — Housing stabilization signals",
            "",
            "### What we verify",
            "- Breadth/dispersion and regime-style stabilization signals using city-level price indices (new & existing homes).",
            "",
            "### Key artifacts",
            *bullets,
            "",
            "> Note: Fallback section auto-generated because `report/sections/property.md` was not found.",
            "",
        ]
    )


def _collect_figs(figs_dir: Path, prefix: str) -> list[Path]:
    """
    Collect figures by prefix from report/figures.
    e.g. prefix='property' -> report/figures/property_*.png
    """
    if not figs_dir.exists():
        return []
    return sorted(figs_dir.glob(f"{prefix}_*.png"))


def _emit_fig_gallery(report_dir: Path, figs: list[Path], title: str) -> list[str]:
    """
    Emit a simple gallery section with embedded images.
    IMPORTANT: paths are relative to report/main.md (which lives in report/).
    So we reference as figures/<filename>.png
    """
    out: list[str] = []
    out += [f"### Figures — {title}", ""]
    if not figs:
        out += ["- (No figures found.)", ""]
        return out

    for p in figs:
        # p is report/figures/xxx.png; main.md is report/main.md
        rel = p.relative_to(report_dir)  # -> figures/xxx.png
        rel_posix = rel.as_posix()

        # Make a clean caption from filename (drop prefix_)
        stem = p.stem
        # remove first "<prefix>_" chunk
        if "_" in stem:
            stem = stem.split("_", 1)[1]
        caption = stem.replace("_", " ").strip()

        out += [
            f"**{caption}**",
            "",
            f"![]({rel_posix})",
            "",
        ]

    return out


def build_main_report(spec: ReportSpec) -> Path:
    logging.info("[Task H] Building main report...")

    repo_root = spec.repo_root
    report_dir = repo_root / spec.report_dir
    figs_dir = repo_root / spec.report_figures_dir
    report_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # 1) Collect figures into one stable location under report/figures
    _copy_figures(repo_root / spec.property_fig_dir, figs_dir, "property")
    _copy_figures(repo_root / spec.external_fig_dir, figs_dir, "external")
    _copy_figures(repo_root / spec.domestic_fig_dir, figs_dir, "domestic")
    _copy_figures(repo_root / spec.model_fig_dir, figs_dir, "model")
    _copy_figures(repo_root / spec.macro_fig_dir, figs_dir, "macro")

    # 2) Read sections (preferred)
    macro_section = _read_if_exists(repo_root / spec.section_macro)
    prop_section = _read_if_exists(repo_root / spec.section_property)
    ext_section = _read_if_exists(repo_root / spec.section_external)
    dom_section = _read_if_exists(repo_root / spec.section_domestic)
    mod_section = _read_if_exists(repo_root / spec.section_model)

    # Backward-compatible fallback: property section
    if not prop_section:
        # legacy report_property if user provided, else auto fallback
        prop_txt = None
        if spec.report_property:
            prop_txt = _read_if_exists(repo_root / spec.report_property)
        prop_section = prop_txt if prop_txt else _property_fallback_section(repo_root)

    # For the other sections: if sections missing, fall back to legacy report_*.md if present
    if not ext_section:
        ext_section = _read_if_exists(repo_root / spec.report_external)
    if not dom_section:
        dom_section = _read_if_exists(repo_root / spec.report_domestic)
    if not mod_section:
        mod_section = _read_if_exists(repo_root / spec.report_model)

    # 3) Collect figure groups
    property_figs = _collect_figs(figs_dir, "property")
    external_figs = _collect_figs(figs_dir, "external")
    domestic_figs = _collect_figs(figs_dir, "domestic")
    model_figs = _collect_figs(figs_dir, "model")
    macro_figs = _collect_figs(figs_dir, "macro")

    # Optional appendix: everything else (should be none if we always prefix)
    all_prefixed = set(macro_figs + property_figs + external_figs + domestic_figs + model_figs)
    appendix_figs = sorted([p for p in figs_dir.glob("*.png") if p not in all_prefixed])

    # 4) Assemble main.md
    main: list[str] = []
    main += [
        "# China Macro Quant Verification — Main Report",
        "",
        "---",
        "",
    ]
    main += ["## Macro Diagnostics", ""]
    if macro_section:
        main += [macro_section.strip(), ""]
    main += _emit_fig_gallery(report_dir, macro_figs, "Macro Diagnostics")
    main += ["", "---", ""]
    # Property
    main += ["## Property", ""]
    if prop_section:
        main += [prop_section.strip(), ""]
    main += _emit_fig_gallery(report_dir, property_figs, "Property")
    main += ["", "---", ""]

    # External
    main += ["## External", ""]
    if ext_section:
        main += [ext_section.strip(), ""]
    else:
        main += ["- Missing `report/sections/external.md` (and legacy `report/report_external.md`).", ""]
    main += _emit_fig_gallery(report_dir, external_figs, "External")
    main += ["", "---", ""]

    # Domestic
    main += ["## Domestic", ""]
    if dom_section:
        main += [dom_section.strip(), ""]
    else:
        main += ["- Missing `report/sections/domestic.md` (and legacy `report/report_domestic.md`).", ""]
    main += _emit_fig_gallery(report_dir, domestic_figs, "Domestic")
    main += ["", "---", ""]

    # Model
    main += ["## Model", ""]
    if mod_section:
        main += [mod_section.strip(), ""]
    else:
        main += ["- Missing `report/sections/model.md` (and legacy `report/report_model.md`).", ""]
    main += _emit_fig_gallery(report_dir, model_figs, "Model")
    main += ["", "---", ""]

    # Appendix (optional)
    if appendix_figs:
        main += ["## Appendix — Additional figures", ""]
        main += _emit_fig_gallery(report_dir, appendix_figs, "Appendix")
        main += ["", "---", ""]


    out_path = report_dir / "main.md"
    out_path.write_text("\n".join(main).rstrip() + "\n", encoding="utf-8")
    logging.info(f"[Task H] Wrote: {out_path}")
    return out_path


def write_readme_pitch(repo_root: Path) -> Path:
    txt = "\n".join(
        [
            "# README_pitch — How to pitch this repo in 2 minutes",
            "",
            "## What this is",
            "- A config-driven macro verification engine that turns narrative claims into auditable charts/tables.",
            "",
            "## What I built",
            "- **Property (Task D):** housing breadth + regime proxy.",
            "- **External (Task E):** trade mix decomposition from two-point data.",
            "- **Domestic (Task F):** demand mix dashboard (consumption/investment/fiscal).",
            "- **Model (Task G):** scenario accounting engine + scenario grid.",
            "- **Report (Task H):** report builder assembling everything into `report/main.md`.",
            "",
            "## How to run",
            "```bash",
            "python -m src.run --config config.yaml",
            "```",
            "",
            "## Where to look",
            "- `report/main.md`",
            "- `report/figures/`",
            "- `output/`",
            "",
            "## How to defend limitations (important)",
            "- External sector is **two-point** → we do structural decomposition, not regression.",
            "- Model betas are **explicit assumptions**, not estimated elasticities.",
            "",
        ]
    )
    out = repo_root / "README_pitch.md"
    out.write_text(txt, encoding="utf-8")
    logging.info(f"[Task H] Wrote: {out}")
    return out
