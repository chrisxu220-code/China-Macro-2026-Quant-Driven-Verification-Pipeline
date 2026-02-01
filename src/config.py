from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def _repo_root() -> Path:
    # repo_root/src/config.py -> repo_root
    return Path(__file__).resolve().parents[1]


def _as_path(repo_root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (repo_root / p)


@dataclass(frozen=True)
class AppPaths:
    repo_root: Path
    data_dir: Path
    output_dir: Path
    logs_dir: Path
    figures_dir: Path
    cache_dir: Path

    def ensure(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class AppConfig:
    raw: Dict[str, Any]
    paths: AppPaths

    @property
    def project_name(self) -> str:
        return str(self.raw.get("project", {}).get("name", "Project"))

    @property
    def steps(self) -> list[str]:
        steps = self.raw.get("run", {}).get("steps", [])
        if not isinstance(steps, list):
            raise ValueError("config.yaml: run.steps must be a list")
        return [str(s) for s in steps]

    @property
    def log_level(self) -> str:
        return str(self.raw.get("logging", {}).get("level", "INFO")).upper()

    @property
    def log_to_file(self) -> bool:
        return bool(self.raw.get("logging", {}).get("to_file", True))

    @property
    def log_filename(self) -> str:
        return str(self.raw.get("logging", {}).get("filename", "run.log"))

    @property
    def seed(self) -> int:
        return int(self.raw.get("repro", {}).get("seed", 42))


def load_config(config_path: str | Path) -> AppConfig:
    repo_root = _repo_root()
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = (repo_root / cfg_path).resolve()

    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    paths_raw = raw.get("paths", {})
    if not isinstance(paths_raw, dict):
        raise ValueError("config.yaml: paths must be a mapping")

    paths = AppPaths(
        repo_root=repo_root,
        data_dir=_as_path(repo_root, str(paths_raw.get("data_dir", "data"))),
        output_dir=_as_path(repo_root, str(paths_raw.get("output_dir", "output"))),
        logs_dir=_as_path(repo_root, str(paths_raw.get("logs_dir", "output/logs"))),
        figures_dir=_as_path(repo_root, str(paths_raw.get("figures_dir", "output/figures"))),
        cache_dir=_as_path(repo_root, str(paths_raw.get("cache_dir", "output/cache"))),
    )

    return AppConfig(raw=raw, paths=paths)
