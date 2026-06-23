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

## Guarded SobolevRNA Polish

This fork adds an optional production-safe SobolevRNA post-processing stage after
the five model CSVs are produced and before the final slot overlay is exported.
The original 1st-place slot plan remains the fallback: a polished candidate only
replaces its raw slot when every safety check passes.

The polish step uses the input C1' coordinates directly. It does **not** use the
stochastic `shr_refine_single` path and never adds Gaussian jitter or random
re-initialization. The local objective is the SobolevRNA coarse Hamiltonian

$$
H(X) = E_{\mathrm{bond}}(X) + E_{\mathrm{steric}}(X) +
       E_{\mathrm{DL}}(X; C),
$$

with the radius-of-gyration term disabled during polish (`rg_target = 0`) so the
optimizer removes local roughness without imposing a new global fold. The terms
are

$$
E_{\mathrm{bond}} =
k_{\mathrm{bond}}\sum_{i=1}^{N-1}
\left(\lVert x_{i+1}-x_i\rVert_2 - 5.95\right)^2,
$$

$$
E_{\mathrm{steric}} =
\sum_{j>i+1}
\left[\max\left(0,\sigma_{\mathrm{clash}} -
\sqrt{\lVert x_i-x_j\rVert_2^2 + 10^{-2}}\right)\right]^2,
$$

$$
E_{\mathrm{DL}} =
w_{\mathrm{DL}}\sum_{i,j} C_{ij}
\left[\max\left(0,\lVert x_i-x_j\rVert_2 - 8.0\right)\right]^2.
$$

The gradient is preconditioned in the discrete cosine basis using the Sobolev
$H^1$ resolvent

$$
\widetilde{\nabla H}
= \operatorname{IDCT}_{II}\left(
\frac{\operatorname{DCT}_{II}(\nabla H)_k}{1+\alpha k^2}
\right),
$$

implemented with `jax.scipy.fft.dct` / `jax.scipy.fft.idct` and
`jax_enable_x64=True`. The default production polish is 2000 steps at
`lr = 0.01`, `alpha = 5.0`, `clip = 2.0`, `k_bond = 100.0`,
`sigma_clash = 3.0`, and `w_DL = 2.0`.

### Safety Gate

For each candidate slot, the refined coordinates are accepted only if all checks
pass:

1. Shape matches the raw candidate and all coordinates are finite, non-sentinel,
   and within the existing sanitizer bound.
2. Bond violations do not increase, where a violation is an adjacent C1'
   distance with `abs(d - 5.95) > 2.0 A`.
3. Steric clashes do not increase, using KDTree pairs below `3.0 A` and
   excluding adjacent residues.
4. `H(refined) < H(raw)` under the same contact map.
5. `Rg(refined)` lies in `[0.7, 1.5] * (3.5 * N**0.45)`.
6. `max(||x[i+1] - x[i]||) < 12.0 A`.
7. `tm_self >= 0.85`, computed by Kabsch-aligning refined to raw coordinates and
   applying a TM-style C1' similarity over valid residues.

Accepted slots are written into `/kaggle/working/submission.csv`. Rejected slots
remain byte-for-byte the original model coordinates for that slot.

### Runtime Controls

| Variable | Default | Effect |
|---|---:|---|
| `SOBOLERNA_POLISH` | `1` | Set to `0` to disable polish and preserve the original ensemble exactly |
| `SOBOLERNA_POLISH_SLOTS` | all | Optional comma-separated slot allowlist, for example `1,2,3` |
| `SOBOLERNA_POLISH_STEPS` | `2000` | Number of Sobolev polish steps |
| `SOBOLERNA_POLISH_LR` | `0.01` | Sobolev polish learning rate |

The stage emits `/kaggle/working/sobolev_polish_report.csv` with per-candidate
metrics and `/kaggle/working/sobolev_polished_slots.csv` with accepted slots.
Docker/SageMaker artifact export treats accepted polished slots as C1'-only
fallback structures because the polish is intentionally C1'-level in v1.

---

## Repository Structure

```
├── solution.ipynb              # Clean solution notebook (submit this to Kaggle)
├── original_submission.ipynb  # Original competition submission (unmodified)
├── sobolev_polish_gate.py      # Guarded SobolevRNA C1' polish and accept gate
├── tests/                      # Unit tests for the polish safety gate
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
| `sobolev_polish_report.csv` | Per-candidate polish metrics and reject reasons |
| `sobolev_polished_slots.csv` | Accepted polished slots used during artifact export |
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
