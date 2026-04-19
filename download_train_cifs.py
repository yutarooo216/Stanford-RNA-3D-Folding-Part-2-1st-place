#!/usr/bin/env python3
"""Download training structure CIF files from RCSB for TBM full-atom output."""
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

csv = "/kaggle/input/stanford-rna-3d-folding-2/train_sequences.csv"
if not os.path.exists(csv):
    print("train_sequences.csv not found, skipping CIF download")
    sys.exit(0)

out_dir = Path("/kaggle/input/train_structures")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv)
pdb_ids = df["target_id"].str[:4].str.upper().unique()
print(f"Downloading {len(pdb_ids)} training CIF files from RCSB...")

failed = []
for pid in sorted(pdb_ids):
    dest = out_dir / f"{pid}.cif"
    if dest.exists() and dest.stat().st_size > 100:
        continue
    url = f"https://files.rcsb.org/download/{pid}.cif"
    r = subprocess.run(
        ["wget", "-q", "--timeout=30", "--tries=3", "-O", str(dest), url],
        capture_output=True,
    )
    if r.returncode != 0 or not dest.exists() or dest.stat().st_size < 100:
        dest.unlink(missing_ok=True)
        failed.append(pid)

ok = len(pdb_ids) - len(failed)
print(f"Done. Downloaded: {ok}/{len(pdb_ids)}, failed: {len(failed)}")
if failed:
    print("Failed IDs (first 20):", failed[:20])
