#!/usr/bin/env bash
# ============================================================
# run_solution.sh — container entrypoint
# Usage:
#   docker run --gpus all \
#     -v /host/my_sequences.csv:/input/test_sequences.csv \
#     -v /host/MSA:/input/MSA \
#     -v /host/output:/kaggle/working/structures \
#     rna3d-solution
#
# Environment variables (all optional):
#   TEST_CSV          path to test_sequences.csv inside container
#                     default: /kaggle/input/stanford-rna-3d-folding-2/test_sequences.csv
#   MSA_DIR           path to MSA directory
#                     default: /kaggle/input/stanford-rna-3d-folding-2/MSA
#   STRUCTURES_OUT    output directory for all-atom CIF/PDB files
#                     default: /kaggle/working/structures
#   NUM_GPUS          number of GPUs to use (auto-detected if unset)
#   BOLTZ_DEVICES     GPUs for Boltz2 inference (defaults to NUM_GPUS)
# ============================================================
set -euo pipefail

WORKING=/kaggle/working
INPUT=/kaggle/input/stanford-rna-3d-folding-2

echo "=== Stanford RNA 3D Folding 2 — Solution Runner ==="
date
echo ""

# ---- GPU detection ----
if [ -z "${NUM_GPUS:-}" ]; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l || echo 1)
    NUM_GPUS=$(( NUM_GPUS > 0 ? NUM_GPUS : 1 ))
fi
export NUM_GPUS
echo "GPUs available: $NUM_GPUS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

# ---- Resolve paths ----
export TEST_CSV="${TEST_CSV:-$INPUT/test_sequences.csv}"
export MSA_DIR="${MSA_DIR:-$INPUT/MSA}"
export STRUCTURES_OUT="${STRUCTURES_OUT:-$WORKING/structures}"
export BOLTZ_DEVICES="${BOLTZ_DEVICES:-$NUM_GPUS}"

# If user bind-mounted custom input, symlink into the expected Kaggle path
if [ -f "/input/test_sequences.csv" ] && [ ! -f "$TEST_CSV" ]; then
    ln -sf /input/test_sequences.csv "$TEST_CSV"
fi
if [ -d "/input/MSA" ] && [ ! -d "$MSA_DIR" ]; then
    ln -sf /input/MSA "$MSA_DIR"
fi

echo "TEST_CSV:        $TEST_CSV"
echo "MSA_DIR:         $MSA_DIR"
echo "STRUCTURES_OUT:  $STRUCTURES_OUT"
echo "BOLTZ_DEVICES:   $BOLTZ_DEVICES"
echo ""

if [ ! -f "$TEST_CSV" ]; then
    echo "ERROR: TEST_CSV not found: $TEST_CSV"
    echo "Bind-mount your sequences file, e.g.:"
    echo "  -v /host/test_sequences.csv:/input/test_sequences.csv"
    exit 1
fi

mkdir -p "$STRUCTURES_OUT"

# ---- Set Kaggle scoring flag so notebook runs in full inference mode ----
export KAGGLE_IS_COMPETITION_RERUN=1

# ---- Run the notebook ----
OUTPUT_NB="$WORKING/solution_output.ipynb"
echo "=== Executing solution notebook ==="

papermill \
    /kaggle/working/solution.ipynb \
    "$OUTPUT_NB" \
    --kernel python3 \
    --execution-timeout 86400 \
    --log-output \
    2>&1 | tee "$WORKING/run.log"

EXIT_CODE=${PIPESTATUS[0]}
echo ""
echo "=== Notebook finished (exit code: $EXIT_CODE) ==="

# ---- Summarise outputs ----
echo ""
echo "--- All-atom structures ---"
if [ -d "$STRUCTURES_OUT" ]; then
    find "$STRUCTURES_OUT" -name "*.cif" -o -name "*.pdb" 2>/dev/null | sort | head -60
    echo "Total: $(find "$STRUCTURES_OUT" -name "*.cif" -o -name "*.pdb" 2>/dev/null | wc -l) files"
else
    echo "(none)"
fi

if [ -f "$WORKING/submission.csv" ]; then
    echo ""
    echo "--- C1' CSV (legacy) ---"
    echo "$WORKING/submission.csv"
fi

exit $EXIT_CODE
