from __future__ import annotations

from pathlib import Path
import yaml
import pandas as pd


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(config: dict) -> None:
    for key, value in config.get("paths", {}).items():
        if key.endswith("_dir") or key in {"results_dir", "logs_dir"}:
            Path(value).mkdir(parents=True, exist_ok=True)


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
