"""
Entrypoint for kaggle1-cp (Stanford RNA 3D Folding 1st place) SageMaker Processing Job.

SageMaker mounts:
  /opt/ml/processing/input/msa        MSA files + protenix_input.json
  /opt/ml/processing/input/kaggle1-cp/datasets
                                      Kaggle datasets tree
  /opt/ml/processing/input/kaggle1-cp/models
                                      Kaggle models tree
  /opt/ml/processing/input/kaggle-input-ref/stanford-rna-3d-folding-2
                                      Kaggle competition input tree
  /opt/ml/processing/input/templates  (not used by this server)
  /opt/ml/processing/output           write final_multichain_models/ here

This entrypoint:
  1. Reads target info from protenix_input.json in the MSA mount.
  2. Constructs /kaggle/input/stanford-rna-3d-folding-2/ directory structure.
  3. Runs solution.ipynb via papermill (full all-atom CIF/PDB output).
  4. Collects up to 5 model files from
     /opt/ml/processing/output/final_submission_artifacts/<target>/slot_*_*,
     preferring PDB over CIF within each slot.
  5. Relabels those models to multi-chain PDBs when needed and writes them to
     /opt/ml/processing/output/final_multichain_models/<target>/.

Environment variables:
  PROCESSING_INPUT_DIR  override input root directory
  INPUT_MSA_DIR   override input MSA directory
  OUTPUT_DIR      override output directory
  SERVER_NAME     CASP server name for PFRMAT TS header
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from relabel_concat_pdb import TooManyChainsError, build_chain_assignments, relabel_pdb

PROCESSING_INPUT_DIR = os.environ.get(
    "PROCESSING_INPUT_DIR", "/opt/ml/processing/input"
)
INPUT_MSA_DIR = os.environ.get("INPUT_MSA_DIR", "/opt/ml/processing/input/msa")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/opt/ml/processing/output")
SERVER_NAME = os.environ.get("SERVER_NAME", "KAGGLE1_TEAM_CP_HHMI")
FINAL_MODELS_DIRNAME = "final_multichain_models"

KAGGLE_INPUT_ROOT = "/kaggle/input"
KAGGLE_INPUT = f"{KAGGLE_INPUT_ROOT}/stanford-rna-3d-folding-2"
KAGGLE_WORKING = "/kaggle/working"
NOTEBOOK_SRC = "/kaggle/working/solution.ipynb"
NOTEBOOK_OUT = f"{KAGGLE_WORKING}/solution_output.ipynb"
DATASETS_INPUT_DIR = Path(PROCESSING_INPUT_DIR) / "kaggle1-cp" / "datasets"
MODELS_INPUT_DIR = Path(PROCESSING_INPUT_DIR) / "kaggle1-cp" / "models"
COMPETITION_INPUT_DIR = (
    Path(PROCESSING_INPUT_DIR) / "kaggle-input-ref" / "stanford-rna-3d-folding-2"
)
# ---------------------------------------------------------------------------
# Arena reconstruction config
# ---------------------------------------------------------------------------

# Arena binary path.
# Example:
#   export ARENA_BIN=/opt/ml/processing/input/arena/Arena
# or place Arena under /app/Arena/Arena or /kaggle/working/Arena/Arena.
ARENA_BIN = os.environ.get("ARENA_BIN", "")

ARENA_WORK_DIRNAME = "arena_reconstructed"

PDB_CHAIN_IDS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
O2_PRIME_NAMES = {"O2'", "O2*"}

# ---------------------------------------------------------------------------
# Target info extraction from protenix_input.json
# ---------------------------------------------------------------------------


def read_protenix_json(msa_dir):
    path = os.path.join(msa_dir, "protenix_input.json")
    with open(path) as f:
        return json.load(f)[0]


def read_protenix_input(msa_dir):
    path = os.path.join(msa_dir, "protenix_input.json")
    with open(path) as f:
        return json.load(f)


def seed_top_level_kaggle_inputs():
    input_root = Path(KAGGLE_INPUT_ROOT)
    input_root.mkdir(parents=True, exist_ok=True)

    if not DATASETS_INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Missing required Kaggle input directory: {DATASETS_INPUT_DIR}"
        )
    datasets_dst = input_root / "datasets"
    if datasets_dst.is_symlink():
        if datasets_dst.resolve() != DATASETS_INPUT_DIR.resolve():
            datasets_dst.unlink()
            datasets_dst.symlink_to(DATASETS_INPUT_DIR, target_is_directory=True)
    elif not datasets_dst.exists():
        datasets_dst.symlink_to(DATASETS_INPUT_DIR, target_is_directory=True)
    else:
        raise FileExistsError(f"Cannot replace existing path: {datasets_dst}")

    if not MODELS_INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Missing required Kaggle input directory: {MODELS_INPUT_DIR}"
        )

    models_dst = input_root / "models"
    if models_dst.is_symlink():
        if models_dst.resolve() != MODELS_INPUT_DIR.resolve():
            models_dst.unlink()
            models_dst.symlink_to(MODELS_INPUT_DIR, target_is_directory=True)
    elif not models_dst.exists():
        models_dst.symlink_to(MODELS_INPUT_DIR, target_is_directory=True)
    else:
        raise FileExistsError(f"Cannot replace existing path: {models_dst}")


def build_competition_view(dataset_dir):
    source_dir = COMPETITION_INPUT_DIR
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Missing required competition input directory: {source_dir}"
        )

    for child in source_dir.iterdir():
        if child.name in {"test_sequences.csv", "MSA"}:
            continue

        dst = Path(dataset_dir) / child.name
        if dst.is_symlink():
            if dst.resolve() == child.resolve():
                continue
            dst.unlink()
        elif dst.exists():
            continue
        dst.symlink_to(child, target_is_directory=child.is_dir())


def require_input_file(path):
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Missing required input file: {file_path}")
    return file_path


def stage_test_sequences(msa_dir, dataset_dir):
    src = require_input_file(Path(msa_dir) / "test_sequences.csv")
    dst = Path(dataset_dir) / "test_sequences.csv"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def stage_flat_msa_files(msa_dir, msa_dst):
    msa_root = Path(msa_dir)
    staged = False
    for src in sorted(msa_root.glob("*.MSA.fasta")):
        staged = True
        dst = Path(msa_dst) / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
    if not staged:
        raise FileNotFoundError(f"No '*.MSA.fasta' files found in {msa_root}")


def setup_kaggle_input(msa_dir):
    seed_top_level_kaggle_inputs()
    os.makedirs(KAGGLE_INPUT, exist_ok=True)

    build_competition_view(KAGGLE_INPUT)
    stage_test_sequences(msa_dir, KAGGLE_INPUT)

    msa_dst = f"{KAGGLE_INPUT}/MSA"
    if os.path.lexists(msa_dst):
        if os.path.islink(msa_dst) or os.path.isfile(msa_dst):
            os.unlink(msa_dst)
        else:
            shutil.rmtree(msa_dst)
    os.makedirs(msa_dst, exist_ok=True)
    stage_flat_msa_files(msa_dir, msa_dst)


# ---------------------------------------------------------------------------
# Kaggle output export
# ---------------------------------------------------------------------------


def copy_if_exists(src, dst):
    src_path = Path(src)
    if not src_path.exists():
        return

    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if src_path.is_dir():
        if dst_path.exists():
            shutil.rmtree(dst_path)
        shutil.copytree(src_path, dst_path)
    else:
        shutil.copy2(src_path, dst_path)


def read_csv_rows(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def rows_by_target(rows):
    grouped = {}
    for row in rows:
        target_id = row["ID"].rsplit("_", 1)[0]
        grouped.setdefault(target_id, []).append(row)
    for target_rows in grouped.values():
        target_rows.sort(key=lambda row: int(row["resid"]))
    return grouped


def load_test_sequences():
    path = Path(KAGGLE_INPUT) / "test_sequences.csv"
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            row["sequence_len"] = len(row["sequence"])
            rows.append(row)
    return rows


def slot_plan_for_length(seq_len):
    if seq_len < 250:
        return {
            1: ("boltz", 1),
            2: ("boltz", 2),
            3: ("rnapro", 1),
            4: ("protenix", 1),
            5: ("drfold", 1),
        }
    if seq_len < 1000:
        return {
            1: ("tbm", 1),
            2: ("boltz", 1),
            3: ("rnapro", 1),
            4: ("rnapro", 2),
            5: ("boltz", 2),
        }
    return {
        1: ("tbm", 1),
        2: ("tbm", 2),
        3: ("tbm", 3),
        4: ("protenix", 1),
        5: ("protenix", 2),
    }


def ranked_boltz_models(target_id):
    target_dir = (
        Path(KAGGLE_WORKING)
        / "boltz_repeat_0"
        / f"boltz_results_{target_id}"
        / "predictions"
        / target_id
    )
    ranked = []
    for idx in range(10):
        json_path = target_dir / f"confidence_{target_id}_model_{idx}.json"
        if not json_path.exists():
            continue
        with json_path.open() as handle:
            data = json.load(handle)
        ranked.append(
            {
                "idx": idx,
                "plddt": data.get("complex_plddt", float("-inf")),
                "conf": data.get("confidence_score", float("-inf")),
            }
        )
    ranked.sort(key=lambda item: (item["plddt"], item["conf"]), reverse=True)
    return ranked

# ---------------------------------------------------------------------------
# Arena reconstruction helpers
# ---------------------------------------------------------------------------

def find_arena_binary():
    """
    Resolve Arena executable.

    Priority:
      1. ARENA_BIN env var
      2. common local locations
      3. PATH
    """
    candidates = []

    if ARENA_BIN:
        candidates.append(Path(ARENA_BIN))

    candidates.extend(
        [
            Path("/app/Arena/Arena"),
            Path("/kaggle/working/Arena/Arena"),
            Path("/opt/ml/processing/input/arena/Arena"),
            Path("/opt/ml/processing/input/Arena/Arena"),
        ]
    )

    for path in candidates:
        if path.exists() and os.access(path, os.X_OK):
            return path

    arena_from_path = shutil.which("Arena")
    if arena_from_path:
        return Path(arena_from_path)

    raise FileNotFoundError(
        "Arena executable was not found. "
        "Set ARENA_BIN=/path/to/Arena or place Arena at /app/Arena/Arena."
    )


def run_arena(input_pdb, output_pdb, mode):
    """
    Run Arena.

    mode=7:
        For C1'-only TBM/DRFold outputs.
    mode=5:
        For RNAPro after removing O2' atoms.
    """
    input_pdb = Path(input_pdb)
    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    arena_bin = find_arena_binary()
    cmd = [str(arena_bin), str(input_pdb), str(output_pdb), str(mode)]

    result = subprocess.run(
        cmd,
        cwd=str(arena_bin.parent),
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Arena failed.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"returncode: {result.returncode}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    if not output_pdb.exists() or output_pdb.stat().st_size == 0:
        raise RuntimeError(
            "Arena finished without producing a valid output PDB.\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    return output_pdb


def safe_float(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "" or value.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def normalize_rna_resname(resname):
    """
    Normalize RNA residue names for PDB.
    """
    resname = str(resname).strip().upper()

    mapping = {
        "A": "A",
        "C": "C",
        "G": "G",
        "U": "U",
        "DA": "A",
        "DC": "C",
        "DG": "G",
        "DT": "U",
        "ADE": "A",
        "CYT": "C",
        "GUA": "G",
        "URA": "U",
    }
    return mapping.get(resname, resname[:3])


def format_pdb_atom_line(
    serial,
    atom_name,
    resname,
    chain_id,
    resid,
    x,
    y,
    z,
    occupancy=1.00,
    bfactor=0.00,
):
    """
    Format one ATOM line for PDB.

    The generated C1' line is sufficient as coarse input for Arena mode 7.
    """
    atom_name = str(atom_name)
    resname = normalize_rna_resname(resname)
    chain_id = str(chain_id)[0]

    try:
        resid_int = int(float(resid))
    except Exception:
        resid_int = int(serial)

    element = atom_name[0].upper()

    return (
        f"ATOM  {serial:5d} {atom_name:>4s} {resname:>3s} {chain_id:1s}"
        f"{resid_int:4d}    "
        f"{float(x):8.3f}{float(y):8.3f}{float(z):8.3f}"
        f"{occupancy:6.2f}{bfactor:6.2f}          {element:>2s}"
    )


def find_target_entry_from_protenix_input(target_id):
    """
    Try to find the target entry in protenix_input.json.

    This is used only to infer chain lengths for C1'-only PDB generation.
    If the schema is unexpected, the code safely falls back to single-chain A.
    """
    try:
        bare_input = read_protenix_input(INPUT_MSA_DIR)
    except Exception:
        return None

    if isinstance(bare_input, list):
        for entry in bare_input:
            if isinstance(entry, dict) and entry.get("name") == target_id:
                return entry
        return bare_input[0] if bare_input else None

    if isinstance(bare_input, dict):
        if bare_input.get("name") == target_id:
            return bare_input
        return bare_input

    return None


def looks_like_rna_sequence(seq):
    if not isinstance(seq, str):
        return False
    seq = seq.strip().upper().replace("T", "U")
    if not seq:
        return False
    valid = set("ACGUN")
    return sum(ch in valid for ch in seq) / max(len(seq), 1) > 0.8


def extract_rna_sequences_recursive(obj, out=None):
    """
    Extract RNA sequence-like fields from protenix_input.json.

    This is intentionally defensive because CASP wrapper input schemas may vary.
    """
    if out is None:
        out = []

    if isinstance(obj, dict):
        # Common patterns:
        # {"id": "A", "sequence": "..."}
        # {"chain_id": "A", "sequence": "..."}
        # {"rna": {"id": "A", "sequence": "..."}}
        if "sequence" in obj and looks_like_rna_sequence(obj.get("sequence")):
            chain_id = (
                obj.get("id")
                or obj.get("chain_id")
                or obj.get("asym_id")
                or obj.get("name")
                or None
            )
            out.append((chain_id, obj["sequence"]))

        for value in obj.values():
            extract_rna_sequences_recursive(value, out)

    elif isinstance(obj, list):
        for value in obj:
            extract_rna_sequences_recursive(value, out)

    return out


def infer_chain_plan(target_id, target_rows):
    """
    Return [(chain_id, chain_length), ...].

    If chain info cannot be inferred reliably, fall back to single chain A.
    """
    n_res = len(target_rows)
    entry = find_target_entry_from_protenix_input(target_id)
    seqs = extract_rna_sequences_recursive(entry)

    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for chain_id, seq in seqs:
        seq_norm = str(seq).strip().upper().replace("T", "U")
        key = (chain_id, seq_norm)
        if key in seen:
            continue
        seen.add(key)
        unique.append((chain_id, seq_norm))

    # Prefer a multi-chain decomposition whose total length matches target_rows.
    if unique:
        total_len = sum(len(seq) for _, seq in unique)
        if total_len == n_res:
            chain_plan = []
            for i, (chain_id, seq) in enumerate(unique):
                if not chain_id or len(str(chain_id)) != 1:
                    chain_id = PDB_CHAIN_IDS[i]
                chain_plan.append((str(chain_id)[0], len(seq)))
            return chain_plan

        # If one sequence exactly matches the target length, use single chain A.
        for chain_id, seq in unique:
            if len(seq) == n_res:
                chain_id = chain_id if chain_id and len(str(chain_id)) == 1 else "A"
                return [(str(chain_id)[0], n_res)]

    return [("A", n_res)]


def write_c1_only_pdb_from_submission_rows(
    target_rows,
    method_rank,
    output_pdb,
    target_id,
):
    """
    Convert Kaggle-style C1' coordinate columns into a coarse PDB.

    Input:
      target_rows: rows from drfold_submission.csv or pred_tbm.csv for one target
      method_rank: 1-based rank. Uses x_1/y_1/z_1, x_2/y_2/z_2, etc.
      output_pdb: generated C1'-only PDB path
      target_id: CASP target id

    Output:
      PDB containing one C1' atom per residue.
    """
    output_pdb = Path(output_pdb)
    output_pdb.parent.mkdir(parents=True, exist_ok=True)

    x_key = f"x_{method_rank}"
    y_key = f"y_{method_rank}"
    z_key = f"z_{method_rank}"

    if not target_rows:
        raise ValueError(
            f"No submission rows found for target={target_id}, method_rank={method_rank}."
        )

    chain_plan = infer_chain_plan(target_id, target_rows)

    lines = []
    serial = 1
    row_idx = 0

    for chain_id, chain_len in chain_plan:
        for _ in range(chain_len):
            if row_idx >= len(target_rows):
                raise ValueError(
                    f"Chain plan exceeds target rows for {target_id}: "
                    f"row_idx={row_idx}, n_rows={len(target_rows)}."
                )

            row = target_rows[row_idx]

            x = safe_float(row.get(x_key))
            y = safe_float(row.get(y_key))
            z = safe_float(row.get(z_key))

            if x is None or y is None or z is None:
                raise ValueError(
                    f"Missing coordinate in {target_id} {x_key}/{y_key}/{z_key} "
                    f"at row_idx={row_idx}, ID={row.get('ID')}."
                )

            resname = row.get("resname", "N")
            resid = row.get("resid", row_idx + 1)

            lines.append(
                format_pdb_atom_line(
                    serial=serial,
                    atom_name="C1'",
                    resname=resname,
                    chain_id=chain_id,
                    resid=resid,
                    x=x,
                    y=y,
                    z=z,
                )
            )

            serial += 1
            row_idx += 1

        lines.append("TER")

    if row_idx != len(target_rows):
        raise ValueError(
            f"Not all rows consumed for {target_id}: "
            f"used={row_idx}, n_rows={len(target_rows)}."
        )

    lines.append("END")
    output_pdb.write_text("\n".join(lines) + "\n")
    return output_pdb


def reconstruct_c1_only_with_arena(
    target_id,
    method_name,
    method_rank,
    target_rows,
):
    """
    TBM / DRFold:
      Kaggle CSV C1' coordinates -> C1'-only PDB -> Arena mode 7 -> all-atom PDB.
    """
    work_dir = (
        Path(KAGGLE_WORKING)
        / ARENA_WORK_DIRNAME
        / target_id
        / f"{method_name}_{method_rank}"
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    c1_pdb = work_dir / f"{target_id}_{method_name}_{method_rank}_c1_only.pdb"
    arena_pdb = work_dir / f"{target_id}_{method_name}_{method_rank}_arena_mode7.pdb"

    write_c1_only_pdb_from_submission_rows(
        target_rows=target_rows,
        method_rank=method_rank,
        output_pdb=c1_pdb,
        target_id=target_id,
    )

    run_arena(c1_pdb, arena_pdb, mode=7)
    return arena_pdb


def remove_o2prime_atoms_from_pdb_text(pdb_text):
    """
    Remove O2' / O2* atom lines from PDB text.
    """
    kept = []

    for line in pdb_text.splitlines():
        if line.startswith(("ATOM", "HETATM")):
            atom_name = line[12:16].strip()
            if atom_name in O2_PRIME_NAMES:
                continue
        kept.append(line.rstrip())

    if not kept or not any(line.startswith(("ATOM", "HETATM")) for line in kept):
        raise ValueError("No ATOM/HETATM lines remained after removing O2' atoms.")

    if kept[-1] != "END":
        kept.append("END")

    return "\n".join(kept) + "\n"


def strip_rnapro_o2prime_and_reconstruct_with_arena(
    target_id,
    method_rank,
    source_structure_path,
):
    """
    RNAPro:
      all-atom PDB/CIF -> remove O2' atoms -> Arena mode 5 -> repaired all-atom PDB.

    If source is CIF, it is first converted to PDB atoms via read_structure_atoms().
    """
    source_structure_path = Path(source_structure_path)

    work_dir = (
        Path(KAGGLE_WORKING)
        / ARENA_WORK_DIRNAME
        / target_id
        / f"rnapro_{method_rank}"
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    no_o2_pdb = work_dir / f"{target_id}_rnapro_{method_rank}_no_o2prime.pdb"
    arena_pdb = work_dir / f"{target_id}_rnapro_{method_rank}_arena_mode5.pdb"

    atoms = read_structure_atoms(source_structure_path)
    if not atoms:
        raise RuntimeError(f"Could not read RNAPro structure: {source_structure_path}")

    no_o2_pdb.write_text(remove_o2prime_atoms_from_pdb_text(atoms))
    run_arena(no_o2_pdb, arena_pdb, mode=5)

    return arena_pdb

def find_structure_files_for_slot(target_id, method_name, method_rank, target_rows=None):
    working_root = Path(KAGGLE_WORKING)
    files = []
    note = None

    # ------------------------------------------------------------
    # Boltz: already all-atom.
    # Select by descending complex_plddt, then confidence_score.
    # ------------------------------------------------------------
    if method_name == "boltz":
        ranked = ranked_boltz_models(target_id)
        if len(ranked) >= method_rank:
            model_idx = ranked[method_rank - 1]["idx"]
            target_dir = (
                working_root
                / "boltz_repeat_0"
                / f"boltz_results_{target_id}"
                / "predictions"
                / target_id
            )
            for ext in (".pdb", ".cif"):
                path = target_dir / f"{target_id}_model_{model_idx}{ext}"
                if path.exists():
                    files.append(path)

            note = (
                f"Boltz rank {method_rank} selected model_{model_idx} using "
                "descending (complex_plddt, confidence_score)."
            )
        else:
            note = f"Boltz rank {method_rank} was not available."

        return files, note

    # ------------------------------------------------------------
    # TBM / DRFold:
    # CSV C1' coordinates -> C1'-only PDB -> Arena mode 7.
    # ------------------------------------------------------------
    if method_name in {"tbm", "drfold"}:
        if not target_rows:
            note = (
                f"{method_name} rank {method_rank} has no CSV rows; "
                "Arena reconstruction was skipped."
            )
            return [], note

        arena_pdb = reconstruct_c1_only_with_arena(
            target_id=target_id,
            method_name=method_name,
            method_rank=method_rank,
            target_rows=target_rows,
        )

        note = (
            f"{method_name} rank {method_rank} was reconstructed from C1' "
            "coordinates using Arena mode 7."
        )
        return [arena_pdb], note

    # ------------------------------------------------------------
    # Protenix / RNAPro:
    # Locate generated all-atom CIF/PDB files.
    # ------------------------------------------------------------
    preferred_roots = []
    if method_name == "protenix":
        preferred_roots = [working_root / "outputs", working_root / "output"]
    elif method_name == "rnapro":
        preferred_roots = [working_root / "output", working_root / "outputs"]

    sample_idx = method_rank - 1

    for root in preferred_roots:
        prediction_dir = root / target_id / "seed_42" / "predictions"
        for ext in (".cif", ".pdb"):
            path = prediction_dir / f"{target_id}_sample_{sample_idx}{ext}"
            if path.exists():
                files.append(path)

    # ------------------------------------------------------------
    # RNAPro:
    # Remove erroneous O2' atoms and reconstruct with Arena mode 5.
    # Return only the repaired PDB so the final collector picks it.
    # ------------------------------------------------------------
    if method_name == "rnapro":
        if files:
            # Prefer the first found file according to the existing root/ext order.
            source_path = files[0]
            repaired_pdb = strip_rnapro_o2prime_and_reconstruct_with_arena(
                target_id=target_id,
                method_rank=method_rank,
                source_structure_path=source_path,
            )
            note = (
                f"RNAPro slot uses sample_{sample_idx}; O2' atoms were removed "
                "and reconstructed using Arena mode 5."
            )
            return [repaired_pdb], note

        note = f"RNAPro sample_{sample_idx} was not available."
        return [], note

    if method_name == "protenix":
        note = (
            f"protenix slot uses sample_{sample_idx} from the method submission "
            f"columns x_{method_rank}/y_{method_rank}/z_{method_rank}."
        )

    return files, note


def projection_rows_for_slot(target_rows, method_rank):
    projected = []
    x_key = f"x_{method_rank}"
    y_key = f"y_{method_rank}"
    z_key = f"z_{method_rank}"
    for row in target_rows:
        projected.append(
            {
                "ID": row["ID"],
                "resname": row["resname"],
                "resid": row["resid"],
                "x": row.get(x_key, ""),
                "y": row.get(y_key, ""),
                "z": row.get(z_key, ""),
            }
        )
    return projected


def write_slot_fallback(slot_dir, target_rows, method_name, method_rank, source_csv, note):
    fallback_path = slot_dir / "selected_output.txt"
    lines = [
        f"method={method_name}",
        f"method_rank={method_rank}",
        f"source_csv={source_csv}",
    ]
    if note:
        lines.append(f"note={note}")
    lines.append("ID,resname,resid,x,y,z")
    for row in projection_rows_for_slot(target_rows, method_rank):
        lines.append(
            ",".join(
                [
                    row["ID"],
                    row["resname"],
                    str(row["resid"]),
                    str(row["x"]),
                    str(row["y"]),
                    str(row["z"]),
                ]
            )
        )
    fallback_path.write_text("\n".join(lines) + "\n")


def export_final_submission_artifacts():
    output_root = Path(OUTPUT_DIR)
    artifact_root = output_root / "final_submission_artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    method_csvs = {
        "protenix": rows_by_target(read_csv_rows(Path(KAGGLE_WORKING) / "protenix_submission.csv")),
        "drfold": rows_by_target(read_csv_rows(Path(KAGGLE_WORKING) / "drfold_submission.csv")),
        "tbm": rows_by_target(read_csv_rows(Path(KAGGLE_WORKING) / "pred_tbm.csv")),
        "rnapro": rows_by_target(read_csv_rows(Path(KAGGLE_WORKING) / "rnapro_submission.csv")),
        "boltz": rows_by_target(read_csv_rows(Path(KAGGLE_WORKING) / "boltz_submission.csv")),
    }

    manifest = []
    for test_row in load_test_sequences():
        target_id = test_row["target_id"]
        seq_len = test_row["sequence_len"]
        target_dir = artifact_root / target_id
        target_dir.mkdir(parents=True, exist_ok=True)

        plan = slot_plan_for_length(seq_len)
        slot_manifest = {
            "target_id": target_id,
            "sequence_len": seq_len,
            "slots": [],
        }

        for slot_idx in range(1, 6):
            method_name, method_rank = plan[slot_idx]
            slot_dir = target_dir / f"slot_{slot_idx}_{method_name}_{method_rank}"
            slot_dir.mkdir(parents=True, exist_ok=True)

            target_rows = method_csvs.get(method_name, {}).get(target_id, [])
            files, note = find_structure_files_for_slot(
                target_id=target_id,
                method_name=method_name,
                method_rank=method_rank,
                target_rows=target_rows,
            )
            copied_files = []
            for src_path in files:
                dst_path = slot_dir / src_path.name
                shutil.copy2(src_path, dst_path)
                copied_files.append(dst_path.relative_to(output_root).as_posix())

            source_csv = f"{method_name}_submission.csv"
            if method_name == "tbm":
                source_csv = "pred_tbm.csv"
            if copied_files:
                details = [
                    f"method={method_name}",
                    f"method_rank={method_rank}",
                    f"source_csv={source_csv}",
                ]
                if note:
                    details.append(f"note={note}")
                details.extend(f"file={path}" for path in copied_files)
                (slot_dir / "selection.txt").write_text("\n".join(details) + "\n")
            else:
                write_slot_fallback(
                    slot_dir=slot_dir,
                    target_rows=target_rows,
                    method_name=method_name,
                    method_rank=method_rank,
                    source_csv=source_csv,
                    note=note,
                )

            slot_manifest["slots"].append(
                {
                    "slot": slot_idx,
                    "method": method_name,
                    "method_rank": method_rank,
                    "source_csv": source_csv,
                    "structure_files": copied_files,
                    "fallback": not copied_files,
                }
            )

        manifest.append(slot_manifest)
        (target_dir / "manifest.json").write_text(json.dumps(slot_manifest, indent=2) + "\n")

    (artifact_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def export_selected_artifacts():
    output_root = Path(OUTPUT_DIR)
    output_root.mkdir(parents=True, exist_ok=True)

    # Notebook-generated inputs consumed by downstream model sections.
    for rel_path in [
        "protenix_input.json",
        "sample_sequences.csv",
        "inputs",
        "output/input",
    ]:
        copy_if_exists(Path(KAGGLE_WORKING) / rel_path, output_root / rel_path)

    # Model outputs collected into submission-style CSVs and their supporting prediction trees.
    for rel_path in [
        "protenix_submission.csv",
        "protenix_submission.debug.csv",
        "drfold_submission.csv",
        "pred_tbm.csv",
        "rnapro_submission.csv",
        "boltz_submission.csv",
        "submission.csv",
        "structures",
        "output",
    ]:
        copy_if_exists(Path(KAGGLE_WORKING) / rel_path, output_root / rel_path)

    boltz_root = Path(KAGGLE_WORKING) / "boltz_repeat_0"
    if boltz_root.exists():
        for predictions_dir in boltz_root.glob("boltz_results_*/predictions"):
            rel_path = predictions_dir.relative_to(Path(KAGGLE_WORKING))
            copy_if_exists(predictions_dir, output_root / rel_path)

    # Preserve generated structure files without re-exporting static source-tree
    # assets bundled inside the notebook workspace.
    working_root = Path(KAGGLE_WORKING)
    for generated_root in [
        working_root / "structures",
        working_root / "output",
        working_root / "outputs",
        working_root / "boltz_repeat_0",
    ]:
        if not generated_root.exists():
            continue
        for pattern in ("*.pdb", "*.cif"):
            for structure_path in generated_root.rglob(pattern):
                rel_path = structure_path.relative_to(working_root)
                copy_if_exists(structure_path, output_root / rel_path)

    export_final_submission_artifacts()


# ---------------------------------------------------------------------------
# Notebook execution
# ---------------------------------------------------------------------------


def run_notebook(target_id):
    structures_out = f"{KAGGLE_WORKING}/structures"
    os.makedirs(structures_out, exist_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "TEST_CSV": f"{KAGGLE_INPUT}/test_sequences.csv",
            "MSA_DIR": f"{KAGGLE_INPUT}/MSA",
            "STRUCTURES_OUT": structures_out,
            "KAGGLE_IS_COMPETITION_RERUN": "1",
        }
    )
    # Copy app directory to working dir for notebook execution (assumes solution.ipynb expects it at /app/)
    shutil.copytree("/app/", "/kaggle/working/", dirs_exist_ok=True)

    cmd = ["./run_solution.sh"]

    result = subprocess.run(cmd, env=env, check=False, cwd="/kaggle/working/")
    if result.returncode != 0:
        print(f"run_solution.sh exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    export_selected_artifacts()


# ---------------------------------------------------------------------------
# Output collection and format conversion
# ---------------------------------------------------------------------------


def pdb_atoms_only(content):
    return "\n".join(
        ln.rstrip()
        for ln in content.splitlines()
        if ln.startswith(("ATOM", "HETATM", "TER"))
    )


def cif_to_pdb_atoms(cif_path):
    try:
        import gemmi

        st = gemmi.read_structure(str(cif_path))
        st.setup_entities()
        return pdb_atoms_only(st.make_pdb_string())
    except Exception as e:
        print(f"WARNING: gemmi conversion failed for {cif_path}: {e}", file=sys.stderr)
        return None


def read_structure_atoms(path):
    return cif_to_pdb_atoms(path) if path.suffix == ".cif" else pdb_atoms_only(path.read_text())


def collect_final_artifact_paths(target_id):
    target_root = Path(OUTPUT_DIR) / "final_submission_artifacts" / target_id
    if not target_root.exists():
        return []

    models = []
    for slot_idx in range(1, 6):
        slot_dirs = sorted(target_root.glob(f"slot_{slot_idx}_*"))
        if not slot_dirs:
            continue
        slot_dir = slot_dirs[0]
        pdb_paths = sorted(slot_dir.glob("*.pdb"))
        if pdb_paths:
            models.append(pdb_paths[0])
            continue

        cif_paths = sorted(slot_dir.glob("*.cif"))
        if cif_paths:
            models.append(cif_paths[0])
    return models


def needs_relabel(rna_chains):
    return len(rna_chains) > 1


def write_final_model_dir(model_paths, bare_input, target_id):
    final_dir = Path(OUTPUT_DIR) / FINAL_MODELS_DIRNAME / target_id
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    chain_assignments, rna_chains = build_chain_assignments(bare_input)
    if not chain_assignments:
        print("ERROR: no RNA chains found in protenix_input.json", file=sys.stderr)
        sys.exit(1)

    if not needs_relabel(rna_chains):
        for path in model_paths:
            shutil.copy2(path, final_dir / path.name)
        return final_dir

    with tempfile.TemporaryDirectory(prefix="kaggle1-cp-relabel-") as tmpdir:
        for idx, path in enumerate(model_paths, start=1):
            atoms = read_structure_atoms(path)
            if not atoms:
                print(f"ERROR: could not read structure from {path}", file=sys.stderr)
                sys.exit(1)
            input_pdb = Path(tmpdir) / f"model_{idx}.input.pdb"
            output_pdb = Path(tmpdir) / f"model_{idx}.output.pdb"
            input_pdb.write_text(atoms.rstrip() + "\nEND\n")
            try:
                stats = relabel_pdb(input_pdb, output_pdb, chain_assignments)
            except TooManyChainsError as e:
                print(f"ERROR: relabel failed for model {idx}: {e}", file=sys.stderr)
                sys.exit(1)

            for warning in stats["warnings"]:
                print(
                    f"WARNING: model {idx} relabel warning: {warning}",
                    file=sys.stderr,
                )
            out_name = path.with_suffix(".pdb").name
            shutil.copy2(output_pdb, final_dir / out_name)
    return final_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if Path(KAGGLE_WORKING).is_symlink():
        Path(KAGGLE_WORKING).unlink()
    elif Path(KAGGLE_WORKING).exists():
        shutil.rmtree(KAGGLE_WORKING)
    Path(KAGGLE_WORKING).parent.mkdir(parents=True, exist_ok=True)
    os.symlink(tempfile.mkdtemp(prefix="kaggle1-cp-working-"), KAGGLE_WORKING)

    bare_input = read_protenix_input(INPUT_MSA_DIR)
    entry = bare_input[0]
    target_id = entry["name"]
    print(f"Target: {target_id}", flush=True)

    setup_kaggle_input(INPUT_MSA_DIR)
    run_notebook(target_id)

    model_paths = collect_final_artifact_paths(target_id)
    if not model_paths:
        print("ERROR: no model files found in final_submission_artifacts/", file=sys.stderr)
        sys.exit(1)
    final_dir = write_final_model_dir(model_paths, bare_input, target_id)
    print(f"Wrote final models under {final_dir}", flush=True)


if __name__ == "__main__":
    main()
