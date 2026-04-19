# Stanford RNA 3D Folding 2 — 1st Place Solution

**Competition:** [Stanford RNA 3D Folding 2](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2)  
**Task:** Predict 3D coordinates of RNA C1' atoms from sequence  
**Metric:** TM-score (higher is better)  
**Result:** 🥇 1st place  

---

## Approach

Five structure prediction models are run independently, then their outputs are combined into a single 5-prediction submission based on sequence length.

### Models

| Model | Role | Sequence length |
|---|---|---|
| [Boltz2](https://github.com/jwohlwend/boltz) | Deep learning-based structure prediction | seq_len < 250 (base), 250–999 (overlay) |
| [DRFold2](https://github.com/leeyang/DRfold) | Energy-based RNA folding | seq_len < 250 (overlay) |
| TBM (Template-Based Modeling) | Sequence similarity search against training set | seq_len ≥ 250 (base), seq_len ≥ 1000 (base) |
| [RNApro](https://github.com/ml4bio/RNAPro) | Transformer-based structure prediction with templates | seq_len < 1000 (overlay) |
| [Protenix](https://github.com/bytedance/Protenix) | AlphaFold3-based structure prediction | seq_len < 250 (overlay), seq_len ≥ 1000 (overlay) |

### Ensemble Strategy

Each submission requires 5 predicted structures per target. Since scoring is best-of-5, the 5 predictions are composed as follows:

| seq_len | pred 1 | pred 2 | pred 3 | pred 4 | pred 5 |
|---|---|---|---|---|---|
| < 250 | Boltz2₁ | Boltz2₂ | RNApro₁ | Protenix₁ | DRFold2₁ |
| 250 – 999 | TBM₁ | Boltz2₁ | RNApro₁ | RNApro₂ | Boltz2₂ |
| ≥ 1000 | TBM₁ | TBM₂ | TBM₃ | Protenix₁ | Protenix₂ |

Each model is run on sequences it handles best (by length), and the results are overlaid in order of increasing specificity.

---

## Repository Structure

```
├── solution.ipynb              # Clean solution notebook (submit this to Kaggle)
├── original_submission.ipynb  # Original competition submission (unmodified)
├── README.md                   # This file
└── submission_docs.zip         # Winner model submission package (B2–B6)
    ├── README.md
    ├── requirements.txt
    ├── SETTINGS.json
    └── directory_structure.txt
```

- **`solution.ipynb`** — Refactored version: dead code removed, ensemble logic clarified. Produces identical results to the original submission.
- **`original_submission.ipynb`** — The exact notebook submitted to the competition, preserved as-is.
- **`submission_docs.zip`** — Documentation package for winner model submission (B2–B6).

---

## Hardware & Environment

| Item | Spec |
|---|---|
| GPU | NVIDIA Tesla P100 × 1 (16 GB VRAM) |
| CPU | Intel Xeon (4 cores) |
| RAM | 29 GB |
| OS | Ubuntu 20.04 (Kaggle environment) |
| Python | 3.12 |
| CUDA | 11.x (Kaggle default) |

---

## How to Reproduce

This notebook is designed to run on Kaggle with GPU.

The following external datasets must be added to the notebook:

| Dataset | Used by |
|---|---|
| `tobimichigan/biotite-1-2` | Protenix, RNApro |
| `qiweiyin/protenix-v1-adjusted` | Protenix |
| `z1493916656/drfold-model-bf16` | DRFold2 |
| `kami1976/biopython-cp312` | TBM, RNApro, Boltz2 |
| `theoviel/rnapro-src` | RNApro |
| `jaejohn/rnapro-ccd-cache` | RNApro |
| `lbugnon/boltz-src-minimal` | Boltz2 |
| `youhanlee/boltz-dependencies` | Boltz2 |
| `yutaroito/boltz-env-depend` | Boltz2 |
| `yutaroito/rdkit-312` | Boltz2 |
| `lbugnon/boltz2` | Boltz2 (model weights) |
| `models/shujun717/ribonanzanet2` | RNApro |

The competition dataset `stanford-rna-3d-folding-2` must also be attached.

### Steps

1. Open `solution.ipynb` on Kaggle
2. Add all datasets listed above
3. Set accelerator to **GPU P100**
4. Click **Run All**
5. Download `submission.csv` from `/kaggle/working/`

### Training

No training is required. All models use pre-trained weights provided via the Kaggle Datasets listed above.

### Files generated during inference

The following files and directories are created in `/kaggle/working/` during execution:

| Path | Description |
|---|---|
| `protenix_submission.csv` | Protenix predictions |
| `drfold_submission.csv` | DRFold2 predictions |
| `pred_tbm.csv` | TBM predictions |
| `rnapro_submission.csv` | RNApro predictions |
| `boltz_submission.csv` | Boltz2 predictions |
| `submission.csv` | Final ensembled submission |
| `RNAPro/` | RNApro source copied from dataset |
| `chunks/` | Intermediate chunked sequences |
| `inputs/` | Boltz2 input YAML files |
| `output/` | Intermediate model outputs |

### Key assumptions

- The environment variable `KAGGLE_IS_COMPETITION_RERUN` must be set. Without it, the notebook writes an all-zeros submission and exits early.
- All model weights and source code are provided via Kaggle Datasets (no internet access required).
- All I/O paths are defined in `SETTINGS.json`.

---

## CASP17 — Running Outside Kaggle (Docker)

For CASP17 and other external use, a Docker-based setup is provided that bakes all model weights and datasets into the image and accepts arbitrary input sequences.

### Key files

| File | Description |
|---|---|
| `Dockerfile.solution` | Builds the self-contained inference image (downloads all Kaggle datasets at build time) |
| `docker-compose.solution.yml` | Convenience wrapper with GPU reservation and volume mounts |
| `solution_custum.ipynb` | Modified notebook used inside the container: full all-atom CIF/PDB output (5 structures per target), protein–RNA complex support, English comments |
| `run_solution.sh` | Container entrypoint — GPU detection, path resolution, papermill execution |
| `download_train_cifs.py` | Downloads training CIF files from RCSB at build time (required for TBM full-atom output) |

### Prerequisites

- Docker with BuildKit enabled
- NVIDIA Container Toolkit (`nvidia-docker2`)
- Kaggle API credentials at `~/.kaggle/kaggle.json`
- Accepted competition terms for `stanford-rna-3d-folding-2`

### Build

All datasets and model weights are downloaded from Kaggle during the build. Credentials are passed via `--secret` and do **not** end up in any image layer.

```bash
DOCKER_BUILDKIT=1 docker build \
  --secret id=kaggle,src=$HOME/.kaggle/kaggle.json \
  -f Dockerfile.solution \
  -t rna3d-solution:latest \
  .
```

Expected image size: ~50–100 GB. Build time is dominated by Kaggle downloads (~30–60 min).

### Run

**Using competition test sequences baked into the image:**

```bash
docker run --gpus all \
  -v $(pwd)/output:/kaggle/working/structures \
  rna3d-solution:latest
```

**Using custom input sequences (CASP17 targets):**

The input CSV must follow the competition schema: `target_id`, `sequence`, `all_sequences`, `stoichiometry`, `ligand_SMILES`, `ligand_ids`.

```bash
docker run --gpus all \
  -v /path/to/test_sequences.csv:/input/test_sequences.csv:ro \
  -v /path/to/MSA:/input/MSA:ro \
  -v $(pwd)/output:/kaggle/working/structures \
  rna3d-solution:latest
```

**Via docker-compose:**

```bash
TEST_CSV=/path/to/test_sequences.csv \
MSA_DIR=/path/to/MSA \
OUTPUT_DIR=$(pwd)/output \
docker compose -f docker-compose.solution.yml up
```

### GPU requirements

| GPU | VRAM | Notes |
|---|---|---|
| L4 | 24 GB | Default target; set `BOLTZ_DEVICES=1` |
| A100 / RTX PRO 6000 | 80–96 GB | Can use `BOLTZ_DEVICES=2` for speed |

Override GPU count:

```bash
docker run --gpus all -e BOLTZ_DEVICES=2 -v ... rna3d-solution:latest
```

### Output

All-atom structures are written to the mounted output directory:

```
output/
  {target_id}/
    {target_id}_model_1.cif   # best prediction (all-atom)
    {target_id}_model_2.cif
    {target_id}_model_3.cif
    {target_id}_model_4.cif
    {target_id}_model_5.cif
```

Format is CIF for Protenix / RNApro outputs and PDB for Boltz2 / TBM / DRFold2 outputs.
The legacy C1' CSV is also written to `/kaggle/working/submission.csv`.

### Differences from the Kaggle notebook

| Feature | Kaggle (`solution.ipynb`) | CASP17 (`solution_custum.ipynb`) |
|---|---|---|
| Output | C1' coordinates only (`submission.csv`) | Full all-atom structures (CIF/PDB) |
| Protein–RNA complexes | Partial | Full support in all models |
| Input | Competition test set only | Arbitrary sequences via bind-mount |
| BOLTZ_DEVICES | Fixed at 1 | Configurable via env var |

---

## Discussion

→ [Kaggle Discussion Post: 1st Place Solution](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/writeups/1st-place-solution-five-model-ensemble)
