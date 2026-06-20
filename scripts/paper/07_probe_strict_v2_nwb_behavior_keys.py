#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import h5py
import pandas as pd

STRICT_SESSIONS = [
    "715093703",
    "719161530",
    "750749662",
    "751348571",
    "755434585",
    "756029989",
]

KEY_PATTERNS = [
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


def find_nwb(root: Path, sid: str) -> Path | None:
    candidates = []
    for pattern in [
        f"**/*{sid}*.nwb",
        f"**/*{sid}*.h5",
        f"**/*{sid}*.hdf5",
    ]:
        candidates.extend(root.glob(pattern))

    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda p: p.stat().st_size, reverse=True)
    return candidates[0]


def visit_h5(path: Path) -> list[dict]:
    rows = []

    def visitor(name, obj):
        lname = name.lower()
        if any(k in lname for k in KEY_PATTERNS):
            row = {
                "h5_path": name,
                "object_type": type(obj).__name__,
                "shape": "",
                "dtype": "",
                "attrs": "",
            }
            if hasattr(obj, "shape"):
                row["shape"] = str(obj.shape)
            if hasattr(obj, "dtype"):
                row["dtype"] = str(obj.dtype)
            try:
                row["attrs"] = ";".join([str(k) for k in obj.attrs.keys()])
            except Exception:
                row["attrs"] = ""
            rows.append(row)

    with h5py.File(path, "r") as f:
        f.visititems(visitor)

    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="data/raw", help="Root directory to search for cached NWB/HDF5 files.")
    ap.add_argument("--out", default="results/reviewer_tests/allen_spontaneous_strict_v2/nwb_behavior_key_inventory.csv")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for sid in STRICT_SESSIONS:
        nwb = find_nwb(raw_root, sid)
        if nwb is None:
            all_rows.append({
                "session_id": sid,
                "nwb_file": "",
                "file_exists": False,
                "h5_path": "",
                "object_type": "",
                "shape": "",
                "dtype": "",
                "attrs": "",
            })
            continue

        try:
            rows = visit_h5(nwb)
        except Exception as e:
            all_rows.append({
                "session_id": sid,
                "nwb_file": str(nwb),
                "file_exists": True,
                "h5_path": f"ERROR: {e}",
                "object_type": "",
                "shape": "",
                "dtype": "",
                "attrs": "",
            })
            continue

        if not rows:
            all_rows.append({
                "session_id": sid,
                "nwb_file": str(nwb),
                "file_exists": True,
                "h5_path": "",
                "object_type": "",
                "shape": "",
                "dtype": "",
                "attrs": "",
            })
        else:
            for r in rows:
                r["session_id"] = sid
                r["nwb_file"] = str(nwb)
                r["file_exists"] = True
                all_rows.append(r)

    df = pd.DataFrame(all_rows)
    cols = ["session_id", "nwb_file", "file_exists", "h5_path", "object_type", "shape", "dtype", "attrs"]
    df = df[cols]
    df.to_csv(out, index=False)

    print("[ok] wrote", out)
    print(df.to_string(index=False, max_colwidth=120))


if __name__ == "__main__":
    main()
