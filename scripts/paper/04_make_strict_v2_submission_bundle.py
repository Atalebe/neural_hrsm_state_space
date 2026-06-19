#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path


DOI = "10.5281/zenodo.20724425"
ZIP_NAME = "neural_hrsm_spontaneous_memory_strict_v2_submission_bundle.zip"
BUNDLE_ROOT = "neural_hrsm_spontaneous_memory_strict_v2_submission_bundle"

RAW_DATA_NOTE = (
    "Raw Allen NWB/HDF5 files are not included. "
    "The manuscript uses publicly available Allen Visual Coding Neuropixels data. "
    "This bundle contains paper-level derived summaries, scripts, figures, tables, "
    "and source files.\n"
)

BASE_FILES = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    ".zenodo.json",
    "Makefile.strict_v2",
    "run_paper_pipeline_strict_v2.sh",
    "neural_hrsm_spontaneous_memory_strict_v2.pdf",
    "paper/neural_hrsm_spontaneous_memory_strict_v2.tex",
    "paper/references.bib",
]

GLOBS = [
    "paper/tables/*.tex",
    "scripts/allen/*.py",
    "scripts/allen/*.sh",
    "scripts/paper/*.py",
    "results/cross_session/allen_spontaneous_strict_v2/*.csv",
    "results/ablation/allen_spontaneous_strict_v2/*.csv",
    "results/figures/cross_session/allen_spontaneous_strict_v2/*.png",
    "results/figures/cross_session/allen_spontaneous_strict_v2/*.csv",
    "results/figures/ablation/allen_spontaneous_strict_v2/*.png",
    "results/figures/ablation/allen_spontaneous_strict_v2/*.csv",
    "results/figures/manuscript/allen_spontaneous_strict_v2/*.png",
    "results/figures/manuscript/allen_spontaneous_strict_v2/*.csv",
    "results/real_allen/session_*_spontaneous_v1/*.csv",
]

EXCLUDED_DIR_PARTS = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
    "submission_bundle",
    "Logbook in full",
}

EXCLUDED_SUFFIXES = (
    ".aux",
    ".bbl",
    ".blg",
    ".fdb_latexmk",
    ".fls",
    ".log",
    ".out",
    ".synctex.gz",
    ".pyc",
    ".pyo",
    ".nwb",
    ".h5",
    ".hdf5",
    ".part",
    ".fastq",
    ".fastq.gz",
    ".fq",
    ".fq.gz",
    ".bam",
    ".sam",
    ".cram",
)


def run_cmd(cmd: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(
            cmd,
            cwd=cwd,
            stderr=subprocess.STDOUT,
            text=True,
        ).strip()
    except Exception:
        return ""


def repo_root() -> Path:
    out = run_cmd(["git", "rev-parse", "--show-toplevel"], Path.cwd())
    if out:
        return Path(out).resolve()
    return Path.cwd().resolve()


def is_excluded(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    rel_s = rel.as_posix()

    if any(part in EXCLUDED_DIR_PARTS for part in rel.parts):
        return True

    if rel_s.startswith("data/raw/"):
        return True
    if rel_s.startswith("data/interim/"):
        return True
    if rel_s.startswith("data/processed/"):
        return True

    return rel_s.endswith(EXCLUDED_SUFFIXES)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def add_file(
    files: set[Path],
    root: Path,
    rel: str,
    missing_required: list[str],
    missing_referenced: list[str],
    required: bool = False,
    referenced: bool = False,
) -> None:
    p = root / rel
    if p.exists() and p.is_file() and not is_excluded(p, root):
        files.add(p.resolve())
        return

    if required:
        missing_required.append(rel)
    elif referenced:
        missing_referenced.append(rel)


def add_glob(files: set[Path], root: Path, pattern: str) -> None:
    for p in root.glob(pattern):
        if p.exists() and p.is_file() and not is_excluded(p, root):
            files.add(p.resolve())


def referenced_figures(root: Path) -> list[str]:
    tex = root / "paper/neural_hrsm_spontaneous_memory_strict_v2.tex"
    if not tex.exists():
        return []

    text = tex.read_text(errors="replace")
    refs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text)
    return [r.replace("\\_", "_").strip() for r in refs]


def collect_files(root: Path) -> tuple[list[Path], list[str], list[str]]:
    files: set[Path] = set()
    missing_required: list[str] = []
    missing_referenced: list[str] = []

    required_files = {
        "neural_hrsm_spontaneous_memory_strict_v2.pdf",
        "paper/neural_hrsm_spontaneous_memory_strict_v2.tex",
        "paper/references.bib",
    }

    for rel in BASE_FILES:
        add_file(
            files,
            root,
            rel,
            missing_required,
            missing_referenced,
            required=rel in required_files,
        )

    for pattern in GLOBS:
        add_glob(files, root, pattern)

    for rel in referenced_figures(root):
        add_file(
            files,
            root,
            rel,
            missing_required,
            missing_referenced,
            referenced=True,
        )

    out = sorted(files, key=lambda p: p.relative_to(root).as_posix())
    return out, sorted(set(missing_required)), sorted(set(missing_referenced))


def manifest_csv(rows: list[dict[str, str]]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=["path", "size_bytes", "sha256"])
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def bundle_readme(commit: str, status: str, created_utc: str, doi: str) -> str:
    clean_text = "clean" if not status else "dirty; inspect BUNDLE_GIT_STATUS.txt"

    lines = [
        "# Neural HRSM strict-v2 submission bundle",
        "",
        "Created UTC: " + created_utc,
        "",
        "Git commit: `" + (commit or "unknown") + "`",
        "",
        "Working tree status at bundle time: `" + clean_text + "`",
        "",
        "Zenodo DOI: https://doi.org/" + doi,
        "",
        "## Main manuscript files",
        "",
        "- `neural_hrsm_spontaneous_memory_strict_v2.pdf`",
        "- `paper/neural_hrsm_spontaneous_memory_strict_v2.tex`",
        "- `paper/references.bib`",
        "- `paper/tables/`",
        "",
        "## Rebuild manuscript",
        "",
        "From the extracted bundle root:",
        "",
        "    latexmk -pdf -interaction=nonstopmode -halt-on-error paper/neural_hrsm_spontaneous_memory_strict_v2.tex",
        "",
        "To regenerate tables and manuscript source from included derived CSVs:",
        "",
        "    python scripts/paper/01_make_strict_v2_tables.py",
        "    python scripts/paper/02_write_strict_v2_manuscript.py",
        "    latexmk -pdf -interaction=nonstopmode -halt-on-error paper/neural_hrsm_spontaneous_memory_strict_v2.tex",
        "",
        "If running inside the full repository:",
        "",
        "    ./run_paper_pipeline_strict_v2.sh",
        "",
        "## Raw data note",
        "",
        RAW_DATA_NOTE.strip(),
        "",
        "## Integrity files",
        "",
        "- `BUNDLE_MANIFEST.csv`: file paths, sizes, and SHA256 checksums.",
        "- `BUNDLE_SHA256SUMS.txt`: standard SHA256 checksum listing.",
        "- `BUNDLE_METADATA.json`: Git and bundle metadata.",
        "",
    ]

    return "\n".join(lines)


def build_bundle(root: Path, out_zip: Path, doi: str) -> None:
    files, missing_required, missing_referenced = collect_files(root)

    if missing_required:
        raise SystemExit("[error] missing required files: " + ", ".join(missing_required))

    commit = run_cmd(["git", "rev-parse", "HEAD"], root)
    branch = run_cmd(["git", "branch", "--show-current"], root)
    exact_tag = run_cmd(["git", "describe", "--tags", "--exact-match"], root)
    status = run_cmd(["git", "status", "--short"], root)
    remote = run_cmd(["git", "remote", "-v"], root)
    created_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")

    rows: list[dict[str, str]] = []
    for p in files:
        rel = p.relative_to(root).as_posix()
        rows.append(
            {
                "path": rel,
                "size_bytes": str(p.stat().st_size),
                "sha256": sha256_file(p),
            }
        )

    metadata = {
        "bundle_name": out_zip.name,
        "created_utc": created_utc,
        "repository": root.name,
        "git_commit": commit,
        "git_branch": branch,
        "git_exact_tag": exact_tag,
        "git_status_clean": status == "",
        "zenodo_doi": doi,
        "included_file_count": len(rows),
        "missing_referenced_assets": missing_referenced,
        "raw_data_excluded": True,
    }

    out_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for row in rows:
            src = root / row["path"]
            arcname = BUNDLE_ROOT + "/" + row["path"]
            zf.write(src, arcname=arcname)

        zf.writestr(
            BUNDLE_ROOT + "/BUNDLE_README.md",
            bundle_readme(commit, status, created_utc, doi),
        )
        zf.writestr(BUNDLE_ROOT + "/BUNDLE_RAW_DATA_NOTE.txt", RAW_DATA_NOTE)
        zf.writestr(
            BUNDLE_ROOT + "/BUNDLE_METADATA.json",
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        )
        zf.writestr(BUNDLE_ROOT + "/BUNDLE_GIT_STATUS.txt", status + ("\n" if status else ""))
        zf.writestr(BUNDLE_ROOT + "/BUNDLE_GIT_REMOTE.txt", remote + ("\n" if remote else ""))
        zf.writestr(
            BUNDLE_ROOT + "/BUNDLE_MISSING_REFERENCED_ASSETS.txt",
            "\n".join(missing_referenced) + ("\n" if missing_referenced else ""),
        )
        zf.writestr(BUNDLE_ROOT + "/BUNDLE_MANIFEST.csv", manifest_csv(rows))
        zf.writestr(
            BUNDLE_ROOT + "/BUNDLE_SHA256SUMS.txt",
            "".join(r["sha256"] + "  " + r["path"] + "\n" for r in rows),
        )

    side_manifest = out_zip.with_suffix(".manifest.csv")
    side_metadata = out_zip.with_suffix(".metadata.json")

    side_manifest.write_text(manifest_csv(rows))
    side_metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    print("[ok] wrote " + str(out_zip))
    print("[ok] included files: " + str(len(rows)))
    print("[ok] zip size: " + f"{out_zip.stat().st_size / (1024 * 1024):.2f}" + " MiB")
    print("[ok] side manifest: " + str(side_manifest))
    print("[ok] side metadata: " + str(side_metadata))

    if missing_referenced:
        print("[warn] missing referenced figure/assets:")
        for item in missing_referenced:
            print("  - " + item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="submission_bundle")
    parser.add_argument("--bundle-name", default=ZIP_NAME)
    parser.add_argument("--doi", default=DOI)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = repo_root()
    out_zip = root / args.out_dir / args.bundle_name
    build_bundle(root, out_zip, args.doi)


if __name__ == "__main__":
    main()
