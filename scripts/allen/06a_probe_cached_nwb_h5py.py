#!/usr/bin/env python3
"""
Lightweight HDF5 probe for cached Allen Neuropixels NWB.

This avoids AllenSDK/PyNWB full object construction and only inspects group
structure. It is a safety probe before direct low-memory extraction.
"""

from pathlib import Path
import argparse
import h5py


def walk_limited(name, obj):
    depth = name.count("/")
    if depth <= 3:
        if isinstance(obj, h5py.Dataset):
            print(f"DATASET {name} shape={obj.shape} dtype={obj.dtype}")
        else:
            print(f"GROUP   {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nwb", required=True)
    args = parser.parse_args()

    path = Path(args.nwb)
    if not path.exists():
        raise SystemExit(f"Missing NWB file: {path}")

    print(f"[info] opening {path}")
    print(f"[info] size_gb={path.stat().st_size / 1e9:.3f}")

    with h5py.File(path, "r") as f:
        print("[info] top-level keys:")
        for key in f.keys():
            print(" ", key)

        print("\n[info] units keys:")
        if "units" in f:
            for key in f["units"].keys():
                obj = f["units"][key]
                if isinstance(obj, h5py.Dataset):
                    print(f"  {key}: shape={obj.shape} dtype={obj.dtype}")
                else:
                    print(f"  {key}: group")
        else:
            print("  no /units group found")

        print("\n[info] intervals keys:")
        if "intervals" in f:
            for key in f["intervals"].keys():
                print(" ", key)
        else:
            print("  no /intervals group found")

        print("\n[info] limited tree:")
        f.visititems(walk_limited)


if __name__ == "__main__":
    main()
