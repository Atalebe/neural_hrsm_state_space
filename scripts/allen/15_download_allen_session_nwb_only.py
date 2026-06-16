#!/usr/bin/env python3
"""
Download one Allen Visual Coding Neuropixels session NWB without loading it.

This version uses only the Python standard library. It validates byte counts and
supports resume via HTTP Range requests. It will not rename a partial file to
.nwb unless the final size matches the expected server size.
"""

from pathlib import Path
from urllib.request import Request, urlopen
import argparse
import re
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-id", type=int, required=True)
    parser.add_argument("--cache-dir", default="data/raw/allen_neuropixels_cache")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--chunk-mb", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=12)
    parser.add_argument("--sleep-sec", type=float, default=3.0)
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


def request(url, start=None):
    headers = {"User-Agent": "Mozilla/5.0"}
    if start is not None and start > 0:
        headers["Range"] = f"bytes={start}-"
    return Request(url, headers=headers)


def parse_total_from_headers(resp):
    content_range = resp.headers.get("Content-Range") or resp.headers.get("content-range")
    if content_range:
        m = re.search(r"/(\d+)\s*$", content_range)
        if m:
            return int(m.group(1))

    content_length = resp.headers.get("Content-Length") or resp.headers.get("content-length")
    if content_length:
        return int(content_length)

    return 0


def expected_size(url):
    # Ask for one byte. If Range is supported, Content-Range gives the total.
    try:
        with urlopen(request(url, start=0), timeout=60) as resp:
            total = parse_total_from_headers(resp)
            resp.read(1)
            return total
    except Exception:
        return 0


def main():
    args = parse_args()

    session_dir = Path(args.cache_dir) / f"session_{args.session_id}"
    session_dir.mkdir(parents=True, exist_ok=True)

    out_path = session_dir / f"session_{args.session_id}.nwb"
    part_path = session_dir / f"session_{args.session_id}.nwb.part"

    url = retrieve_link(args.session_id)
    total = expected_size(url)

    print(f"[info] session_id={args.session_id}")
    print(f"[info] url={url}")
    print(f"[info] output={out_path}")
    if total:
        print(f"[info] expected_size_gb={total / 1e9:.3f}")

    if args.force:
        out_path.unlink(missing_ok=True)
        part_path.unlink(missing_ok=True)

    if out_path.exists():
        size = out_path.stat().st_size
        if total and size == total:
            print(f"[ok] already complete: {out_path}")
            print(f"[ok] size_gb={size / 1e9:.3f}")
            return
        if total and size < total:
            print(f"[warn] existing .nwb is partial: {size} < {total}; resuming as .part")
            out_path.rename(part_path)
        elif not total:
            print(f"[ok] existing file found but server total unknown: {out_path}")
            print(f"[ok] size_gb={size / 1e9:.3f}")
            return
        else:
            print(f"[warn] existing .nwb size unexpected; restarting")
            out_path.unlink()

    chunk_size = int(args.chunk_mb) * 1024 * 1024

    for attempt in range(1, args.max_attempts + 1):
        start = part_path.stat().st_size if part_path.exists() else 0

        if total and start >= total:
            break

        print(f"[attempt {attempt}/{args.max_attempts}] resume_byte={start}")

        try:
            req = request(url, start=start)
            with urlopen(req, timeout=90) as resp:
                status = getattr(resp, "status", None)

                # If the server ignores Range and returns full content, restart.
                if start > 0 and status == 200:
                    print("[warn] server ignored Range request; restarting from byte 0")
                    part_path.unlink(missing_ok=True)
                    start = 0

                mode = "ab" if start > 0 else "wb"
                seen = start

                with open(part_path, mode) as f:
                    while True:
                        chunk = resp.read(chunk_size)
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

        except Exception as e:
            print(f"[warn] attempt failed: {type(e).__name__}: {e}")

        size = part_path.stat().st_size if part_path.exists() else 0
        if total and size == total:
            break

        if total:
            print(f"[warn] incomplete after attempt {attempt}: {size} / {total} bytes")
        else:
            print(f"[warn] downloaded {size} bytes; total unknown")

        time.sleep(args.sleep_sec)

    final_size = part_path.stat().st_size if part_path.exists() else 0

    if total and final_size != total:
        raise RuntimeError(
            f"Download incomplete after {args.max_attempts} attempts: {final_size} / {total} bytes. "
            f"Partial file preserved at {part_path}"
        )

    if final_size <= 0:
        raise RuntimeError("Download produced an empty file")

    part_path.rename(out_path)

    print(f"[ok] wrote {out_path}")
    print(f"[ok] size_gb={out_path.stat().st_size / 1e9:.3f}")


if __name__ == "__main__":
    main()
