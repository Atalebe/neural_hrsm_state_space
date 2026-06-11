from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_hrsm_region_summary(domain_table: pd.DataFrame, outpath: str | Path) -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    summary = domain_table.groupby("region")[["H", "R", "S", "M"]].median().sort_index()
    ax = summary.plot(kind="bar", figsize=(9, 5))
    ax.set_ylabel("Median z-scored proxy")
    ax.set_title("Neural HRSM regional summary")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def plot_memory_gain(memory_table: pd.DataFrame, outpath: str | Path) -> None:
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    ax = memory_table.plot(x="session_id", y="memory_gain_r2", kind="bar", legend=False, figsize=(7, 4))
    ax.set_ylabel("Lagged prediction gain, ΔR²")
    ax.set_title("Non-Markovian memory gain by session")
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
