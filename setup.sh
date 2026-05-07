#!/usr/bin/env bash
# Run this on the host BEFORE building/running the Docker image.
# Downloads all Kaggle competition data, datasets, and model weights to
# KAGGLE_INPUT (default: ./kaggle/input), which is then bind-mounted into
# the container at /kaggle/input.
#
# Requires ~/.kaggle/kaggle.json (or KAGGLE_USERNAME + KAGGLE_KEY env vars).
set -euo pipefail

KAGGLE_INPUT="${KAGGLE_INPUT:-$(dirname "$0")/kaggle/input}"

echo "Downloading to: $KAGGLE_INPUT"

# ---- Competition data (test_sequences.csv, MSA, sample_submission, etc.) ----
mkdir -p "$KAGGLE_INPUT/stanford-rna-3d-folding-2"
kaggle competitions download -c stanford-rna-3d-folding-2 \
    -p "$KAGGLE_INPUT/stanford-rna-3d-folding-2"
find "$KAGGLE_INPUT/stanford-rna-3d-folding-2" -name "*.zip" -exec unzip -o -q {} -d "$KAGGLE_INPUT/stanford-rna-3d-folding-2" \;
find "$KAGGLE_INPUT/stanford-rna-3d-folding-2" -name "*.zip" -delete

# ---- Python wheel datasets ----
kaggle datasets download kami1976/biopython-cp312 \
    -p "$KAGGLE_INPUT/datasets/kami1976/biopython-cp312" --unzip
kaggle datasets download yutaroito/boltz-env-depend \
    -p "$KAGGLE_INPUT/datasets/yutaroito/boltz-env-depend" --unzip
kaggle datasets download yutaroito/rdkit-312 \
    -p "$KAGGLE_INPUT/datasets/yutaroito/rdkit-312" --unzip
kaggle datasets download tobimichigan/biotite-1-2 \
    -p "$KAGGLE_INPUT/datasets/tobimichigan/biotite-1-2" --unzip
kaggle datasets download youhanlee/boltz-dependencies \
    -p "$KAGGLE_INPUT/datasets/youhanlee/boltz-dependencies" --unzip

# ---- Model source code datasets ----
kaggle datasets download qiweiyin/protenix-v1-adjusted \
    -p "$KAGGLE_INPUT/datasets/qiweiyin/protenix-v1-adjusted" --unzip
kaggle datasets download z1493916656/drfold-model-bf16 \
    -p "$KAGGLE_INPUT/datasets/z1493916656/drfold-model-bf16" --unzip
kaggle datasets download theoviel/rnapro-src \
    -p "$KAGGLE_INPUT/datasets/theoviel/rnapro-src" --unzip
kaggle datasets download jaejohn/rnapro-ccd-cache \
    -p "$KAGGLE_INPUT/datasets/jaejohn/rnapro-ccd-cache" --unzip

# ---- Boltz2 source + cache ----
kaggle datasets download lbugnon/boltz-src-minimal \
    -p "$KAGGLE_INPUT/datasets/lbugnon/boltz-src-minimal" --unzip
kaggle datasets download lbugnon/boltz2 \
    -p "$KAGGLE_INPUT/datasets/lbugnon/boltz2" --unzip

# ---- RibonanzaNet2 model weights ----
mkdir -p "$KAGGLE_INPUT/models/shujun717/ribonanzanet2/pytorch/alpha/1"
kaggle models instances versions download \
    shujun717/ribonanzanet2/pytorch/alpha/1 \
    -p "$KAGGLE_INPUT/models/shujun717/ribonanzanet2/pytorch/alpha/1"

echo "Done. Mount $KAGGLE_INPUT into the container at /kaggle/input."
echo "Example: docker run -v \"\$PWD/kaggle/input:/kaggle/input\" ..."
