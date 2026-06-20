#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import pandas as pd

STRICT_SESSIONS = [
    "715093703",
    "719161530",
    "750749662",
    "751348571",
    "755434585",
    "756029989",
]

BEHAVIOR_PATTERNS = [
    "run",
    "running",
    "speed",
    "velocity",
    "pupil",
    "eye",
    "lick",
    "wheel",
    "face",
    "motion",
    "behavior",
    "arousal",
]


def main() -> None:
    out_dir = Path("results/reviewer_tests/allen_spontaneous_strict_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sid in STRICT_SESSIONS:
        p = Path("results/real_allen") / f"session_{sid}_spontaneous_v1" / "real_neural_hrsm_bin_level_metrics.csv"
        if not p.exists():
            rows.append({
                "session_id": sid,
                "file_exists": False,
                "n_columns": 0,
                "behavior_like_columns": "",
                "can_run_behavior_conditioned_model_from_current_csv": False,
            })
            continue

        cols = list(pd.read_csv(p, nrows=1).columns)
        behavior_like = [
            c for c in cols
            if any(pattern in c.lower() for pattern in BEHAVIOR_PATTERNS)
        ]

        # State speed is not a behavioral covariate; remove it from this inventory.
        behavior_like = [
            c for c in behavior_like
            if c.lower() not in {"population_state_speed", "state_speed", "speed"}
        ]

        rows.append({
            "session_id": sid,
            "file_exists": True,
            "n_columns": len(cols),
            "behavior_like_columns": ";".join(behavior_like),
            "can_run_behavior_conditioned_model_from_current_csv": bool(behavior_like),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "behavior_covariate_inventory.csv", index=False)

    print("[ok] wrote", out_dir / "behavior_covariate_inventory.csv")
    print(df.to_string(index=False))

    if not df["can_run_behavior_conditioned_model_from_current_csv"].any():
        print()
        print("[note] No running/pupil/face-motion covariates were found in the current derived HRSM CSVs.")
        print("[note] The behavior-conditioned reviewer control will require a separate NWB/HDF5 extraction step.")


if __name__ == "__main__":
    main()
