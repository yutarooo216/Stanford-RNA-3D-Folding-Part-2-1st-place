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
├── solution.ipynb          # Clean solution notebook (submit this to Kaggle)
└── original_submission.ipynb  # Original competition submission (unmodified)
```

- **`solution.ipynb`** — Refactored version: dead code removed, ensemble logic clarified. Produces identical results to the original submission.
- **`original_submission.ipynb`** — The exact notebook submitted to the competition, preserved as-is.

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

---

## Discussion

→ [Kaggle Discussion Post: 1st Place Solution](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/writeups/1st-place-solution-five-model-ensemble)
