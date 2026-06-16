#!/usr/bin/env python3
"""
Download one Allen Visual Coding Neuropixels session NWB without loading it.

This avoids EcephysProjectCache.get_session_data(), which downloads and then
tries to construct a full PyNWB session object. On this machine that path can
trigger OOM. This script only retrieves the direct EcephysNwb download link and
streams the NWB to:

data/raw/allen_neuropixels_cache/session_<id>/session_<id>.nwb
"""

from pathlib import Path
from urllib.request import Request, urlopen
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", type=int, required=True)
    parser.add_argument(
        "--cache-dir",
        default="data/raw/allen_neuropixels_cache",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--chunk-mb", type=int, default=16)
    return parser.parse_args()


def retrieve_link(session_id):
    from allensdk.brain_observatory.ecephys.ecephys_project_api.utilities import (
        build_and_execute,
    )
    from allensdk.brain_observatory.ecephys.ecephys_project_api.rma_engine import (
        RmaEngine,
    )

    rma_engine = RmaEngine(scheme="http", host="api.brain-map.org")

    well_known_files = build_and_execute(
        (
            "criteria=model::WellKnownFile"
            ",rma::criteria,well_known_file_type[name$eq'EcephysNwb']"
            "[attachable_type$eq'EcephysSession']"
            r"[attachable_id$eq{{session_id}}]"
        ),
        engine=rma_engine.get_rma_tabular,
        session_id=session_id,
    )

    if well_known_files.shape[0] < 1:
        raise RuntimeError(f"No EcephysNwb download link found for session {session_id}")

    download_link = str(well_known_files["download_link"].iloc[0])
    return "http://api.brain-map.org/" + download_link.lstrip("/")


def main():
    args = parse_args()

    session_dir = Path(args.cache_dir) / f"session_{args.session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    out_path = session_dir / f"session_{args.session_id}.nwb"
    part_path = session_dir / f"session_{args.session_id}.nwb.part"

    if out_path.exists() and not args.force:
        print(f"[ok] already exists: {out_path}")
        print(f"[ok] size_gb={out_path.stat().st_size / 1e9:.3f}")
        return

    url = retrieve_link(args.session_id)
    print(f"[info] session_id={args.session_id}")
    print(f"[info] url={url}")
    print(f"[info] output={out_path}")

    if part_path.exists():
        part_path.unlink()

    chunk_size = int(args.chunk_mb) * 1024 * 1024
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})

    with urlopen(req, timeout=60) as r:
        total_header = r.headers.get("Content-Length") or r.headers.get("content-length")
        total = int(total_header) if total_header else 0
        seen = 0

        with open(part_path, "wb") as f:
            while True:
                chunk = r.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                seen += len(chunk)

                if total:
                    pct = 100.0 * seen / total
                    print(
                        f"\r[download] {seen / 1e9:.3f} / {total / 1e9:.3f} GB ({pct:.1f}%)",
                        end="",
                        flush=True,
                    )
                else:
                    print(
                        f"\r[download] {seen / 1e9:.3f} GB",
                        end="",
                        flush=True,
                    )

    print()
    part_path.rename(out_path)

    print(f"[ok] wrote {out_path}")
    print(f"[ok] size_gb={out_path.stat().st_size / 1e9:.3f}")


if __name__ == "__main__":
    main()
