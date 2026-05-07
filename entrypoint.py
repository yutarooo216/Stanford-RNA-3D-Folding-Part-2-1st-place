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


def find_structure_files_for_slot(target_id, method_name, method_rank):
    working_root = Path(KAGGLE_WORKING)
    files = []
    note = None

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

    if method_name in {"protenix", "rnapro"}:
        note = (
            f"{method_name} slot uses sample_{sample_idx} from the method submission "
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
            files, note = find_structure_files_for_slot(target_id, method_name, method_rank)
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
