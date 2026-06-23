"""Guarded SobolevRNA polish for 1st-place RNA 3D candidates.

This module is intentionally self-contained so it can run from the Kaggle
notebooks, the Docker runner, and the SageMaker entrypoints.  It ports the
SobolevRNA polish path, not the stochastic re-seeding refinement path.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

SENTINEL = -2_000_000.0
MAX_ABS_COORD = 1499.0
IDEAL_C1_DISTANCE = 5.95
SOBOLERNA_REPORT = "/kaggle/working/sobolev_polish_report.csv"
SOBOLERNA_SLOTS = "/kaggle/working/sobolev_polished_slots.csv"


@dataclass
class PolishMetrics:
    energy_raw: float = math.nan
    energy_refined: float = math.nan
    bond_raw: int = -1
    bond_refined: int = -1
    clash_raw: int = -1
    clash_refined: int = -1
    rg_refined: float = math.nan
    expected_rg: float = math.nan
    max_step_refined: float = math.nan
    tm_self: float = math.nan


@dataclass
class PolishResult:
    target_id: str
    source: str
    rank: int
    slot: int
    accepted: bool
    reject_reason: str
    metrics: PolishMetrics
    raw_coords: np.ndarray
    refined_coords: np.ndarray

    def report_row(self) -> dict:
        row = {
            "target_id": self.target_id,
            "source": self.source,
            "rank": self.rank,
            "slot": self.slot,
            "accepted": self.accepted,
            "reject_reason": self.reject_reason,
        }
        row.update(asdict(self.metrics))
        return row


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def slot_coord_cols(slot: int) -> list[str]:
    return [f"x_{slot}", f"y_{slot}", f"z_{slot}"]


def slot_plan_for_length(seq_len: int) -> dict[int, tuple[str, int]]:
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


def _target_mask(df, target_id: str):
    return df["ID"].astype(str).str.startswith(f"{target_id}_")


def coords_from_slot(df, target_id: str, slot: int) -> np.ndarray:
    rows = df.loc[_target_mask(df, target_id)].copy()
    if "resid" in rows.columns:
        rows = rows.sort_values("resid")
    cols = slot_coord_cols(slot)
    return rows[cols].to_numpy(dtype=np.float64)


def write_coords_to_slot(df, target_id: str, slot: int, coords: np.ndarray):
    mask = _target_mask(df, target_id)
    target_rows = df.loc[mask].copy()
    if "resid" in target_rows.columns:
        target_rows = target_rows.sort_values("resid")
    if len(target_rows) != len(coords):
        raise ValueError(
            f"{target_id} slot {slot}: {len(coords)} coords for {len(target_rows)} rows"
        )
    cols = slot_coord_cols(slot)
    df.loc[target_rows.index, cols] = coords.astype(float)
    return df


def valid_coord_mask(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=np.float64)
    finite = np.isfinite(arr).all(axis=-1)
    in_range = (np.abs(arr) <= MAX_ABS_COORD).all(axis=-1)
    non_sentinel = (arr > SENTINEL + 1.0).all(axis=-1)
    non_zero = ~np.all(arr == 0.0, axis=-1)
    return finite & in_range & non_sentinel & non_zero


def coords_are_valid(coords: np.ndarray) -> bool:
    arr = np.asarray(coords)
    return arr.ndim == 2 and arr.shape[1] == 3 and bool(valid_coord_mask(arr).all())


def bond_violation_count(
    coords: np.ndarray,
    ideal: float = IDEAL_C1_DISTANCE,
    tolerance: float = 2.0,
) -> int:
    arr = np.asarray(coords, dtype=np.float64)
    if len(arr) < 2:
        return 0
    mask = valid_coord_mask(arr)
    pair_mask = mask[1:] & mask[:-1]
    if not pair_mask.any():
        return 0
    dists = np.linalg.norm(arr[1:] - arr[:-1], axis=1)
    return int(np.sum((np.abs(dists - ideal) > tolerance) & pair_mask))


def steric_clash_count(coords: np.ndarray, clash_distance: float = 3.0) -> int:
    arr = np.asarray(coords, dtype=np.float64)
    mask = valid_coord_mask(arr)
    valid_indices = np.where(mask)[0]
    points = arr[mask]
    if len(points) < 3:
        return 0
    try:
        from scipy.spatial import cKDTree

        pairs = cKDTree(points).query_pairs(clash_distance)
        return int(
            sum(abs(int(valid_indices[i]) - int(valid_indices[j])) > 1 for i, j in pairs)
        )
    except Exception:
        count = 0
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if abs(int(valid_indices[i]) - int(valid_indices[j])) <= 1:
                    continue
                if np.linalg.norm(points[i] - points[j]) < clash_distance:
                    count += 1
        return count


def radius_of_gyration(coords: np.ndarray) -> float:
    arr = np.asarray(coords, dtype=np.float64)
    mask = valid_coord_mask(arr)
    if not mask.any():
        return math.nan
    pts = arr[mask]
    centered = pts - pts.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))


def expected_rg(n_residues: int) -> float:
    return float(3.5 * (n_residues**0.45))


def max_consecutive_distance(coords: np.ndarray) -> float:
    arr = np.asarray(coords, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    mask = valid_coord_mask(arr)
    pair_mask = mask[1:] & mask[:-1]
    if not pair_mask.any():
        return math.inf
    dists = np.linalg.norm(arr[1:] - arr[:-1], axis=1)
    return float(np.max(dists[pair_mask]))


def _kabsch_align(mobile: np.ndarray, target: np.ndarray) -> np.ndarray:
    mob = np.asarray(mobile, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    mob_centroid = mob.mean(axis=0)
    tgt_centroid = tgt.mean(axis=0)
    h = (mob - mob_centroid).T @ (tgt - tgt_centroid)
    u, _s, vt = np.linalg.svd(h)
    if np.linalg.det(vt.T @ u.T) < 0:
        vt[-1, :] *= -1
    rotation = vt.T @ u.T
    return (mob - mob_centroid) @ rotation.T + tgt_centroid


def tm_self_score(raw_coords: np.ndarray, refined_coords: np.ndarray) -> float:
    raw = np.asarray(raw_coords, dtype=np.float64)
    refined = np.asarray(refined_coords, dtype=np.float64)
    if raw.shape != refined.shape or raw.ndim != 2 or raw.shape[1] != 3:
        return 0.0
    mask = valid_coord_mask(raw) & valid_coord_mask(refined)
    if int(mask.sum()) < 3:
        return 0.0
    raw_pts = raw[mask]
    refined_aligned = _kabsch_align(refined[mask], raw_pts)
    dists = np.linalg.norm(refined_aligned - raw_pts, axis=1)
    n = len(dists)
    d0 = max(0.5, 1.24 * np.cbrt(max(n - 15, 1)) - 1.8)
    return float(np.mean(1.0 / (1.0 + (dists / d0) ** 2)))


def _try_call_contact_provider(
    contact_provider: Callable | None,
    sequence: str,
    target_id: str,
    context: Mapping | None,
):
    if contact_provider is None:
        return None
    context = context or {}
    attempts = [
        (
            sequence,
            target_id,
            context.get("train_coords_dict"),
            context.get("train_seqs_df"),
            context.get("segments_map"),
        ),
        (sequence, target_id),
        (sequence,),
    ]
    for args in attempts:
        try:
            return contact_provider(*args)
        except TypeError:
            continue
    return None


def _segments_for_target(segments_map, target_id: str):
    if not segments_map:
        return None
    try:
        return segments_map.get(target_id)
    except AttributeError:
        return None


def maybe_block_diagonal_contacts(
    contact_map: np.ndarray,
    n_residues: int,
    segments: Sequence[tuple[int, int]] | None,
) -> np.ndarray:
    cmap = np.asarray(contact_map, dtype=np.float32)
    if cmap.shape == (n_residues, n_residues):
        return cmap
    if not segments or len(segments) < 2:
        return np.zeros((n_residues, n_residues), dtype=np.float32)
    lengths = [int(e) - int(s) for s, e in segments]
    if len(set(lengths)) != 1 or cmap.shape != (lengths[0], lengths[0]):
        return np.zeros((n_residues, n_residues), dtype=np.float32)
    full = np.zeros((n_residues, n_residues), dtype=np.float32)
    for start, end in segments:
        start = int(start)
        end = int(end)
        full[start:end, start:end] = cmap[: end - start, : end - start]
    return full


def build_contact_map(
    sequence: str,
    target_id: str,
    contact_provider: Callable | None = None,
    segments_map=None,
    context: Mapping | None = None,
) -> np.ndarray:
    n_residues = len(sequence)
    cmap = _try_call_contact_provider(contact_provider, sequence, target_id, context)
    if cmap is None:
        return np.zeros((n_residues, n_residues), dtype=np.float32)
    segments = _segments_for_target(segments_map, target_id)
    return maybe_block_diagonal_contacts(cmap, n_residues, segments)


_JAX_ENGINE = None


def _load_jax_engine():
    global _JAX_ENGINE
    if _JAX_ENGINE is not None:
        return _JAX_ENGINE

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import jax.scipy.fft as jfft

    def energy_bond(coords, d0=5.95, k_bond=100.0):
        diffs = coords[1:] - coords[:-1]
        dists = jnp.sqrt(jnp.sum(diffs**2, axis=-1) + 1e-8)
        return k_bond * jnp.sum((dists - d0) ** 2)

    def energy_steric(coords, sigma_clash=3.0):
        diff = coords[:, None, :] - coords[None, :, :]
        dist = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-2)
        n = coords.shape[0]
        i_idx = jnp.arange(n)
        mask = i_idx[None, :] > i_idx[:, None] + 1
        violations = jnp.maximum(sigma_clash - dist, 0.0)
        return jnp.sum((violations * mask) ** 2)

    def energy_contacts(coords, contact_map, d_contact=8.0, w_dl=2.0):
        diff = coords[:, None, :] - coords[None, :, :]
        dists = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-8)
        violations = jnp.maximum(dists - d_contact, 0.0)
        return w_dl * jnp.sum(contact_map * (violations**2))

    def energy_rg(coords, rg_target, k_rg=1.0):
        center = jnp.mean(coords, axis=0)
        rg_sq = jnp.mean(jnp.sum((coords - center) ** 2, axis=-1))
        rg = jnp.sqrt(rg_sq + 1e-8)
        violation = jnp.maximum(rg_target - rg, 0.0)
        return k_rg * (violation**2)

    def total_energy(coords, contact_map, params):
        return (
            energy_bond(coords, k_bond=params.get("k_bond", 100.0))
            + energy_steric(coords, sigma_clash=params.get("sigma_clash", 3.0))
            + energy_contacts(coords, contact_map, w_dl=params.get("w_DL", 2.0))
            + energy_rg(coords, params.get("rg_target", 0.0), k_rg=1.0)
        )

    def sobolev_h1_smooth(gradient, alpha=5.0):
        n = gradient.shape[0]
        k = jnp.arange(n)
        inv_sigma = 1.0 / (1.0 + alpha * k**2)
        g_hat = jfft.dct(gradient, type=2, norm="ortho", axis=0)
        return jfft.idct(g_hat * inv_sigma[:, None], type=2, norm="ortho", axis=0)

    @jax.jit
    def step_fn(coords, params):
        grads = jax.grad(total_energy)(coords, params["contact_map"], params)
        smooth_grads = sobolev_h1_smooth(grads, alpha=params["alpha"])
        clipped_grads = jnp.clip(smooth_grads, -params["clip"], params["clip"])
        return coords - params["lr"] * clipped_grads

    def shr_polish(coords, contact_map=None, n_steps=2000, lr=0.01):
        coords_jax = jnp.array(coords, dtype=jnp.float64)
        if contact_map is None:
            contact_map_jax = jnp.zeros((len(coords), len(coords)), dtype=jnp.float64)
        else:
            contact_map_jax = jnp.array(contact_map, dtype=jnp.float64)
        params = {
            "lr": lr,
            "alpha": 5.0,
            "clip": 2.0,
            "w_DL": 2.0,
            "k_bond": 100.0,
            "sigma_clash": 3.0,
            "rg_target": 0.0,
            "contact_map": contact_map_jax,
        }

        def scan_body(carrier, _):
            return step_fn(carrier, params), None

        final_coords, _ = jax.lax.scan(scan_body, coords_jax, None, length=n_steps)
        return np.array(final_coords)

    def energy_value(coords, contact_map=None):
        coords_jax = jnp.array(coords, dtype=jnp.float64)
        if contact_map is None:
            contact_map_jax = jnp.zeros((len(coords), len(coords)), dtype=jnp.float64)
        else:
            contact_map_jax = jnp.array(contact_map, dtype=jnp.float64)
        params = {
            "w_DL": 2.0,
            "k_bond": 100.0,
            "sigma_clash": 3.0,
            "rg_target": 0.0,
        }
        return float(total_energy(coords_jax, contact_map_jax, params))

    _JAX_ENGINE = {"polisher": shr_polish, "energy_fn": energy_value}
    return _JAX_ENGINE


def jax_polisher(raw_coords: np.ndarray, contact_map: np.ndarray | None) -> np.ndarray:
    n_steps = int(os.environ.get("SOBOLERNA_POLISH_STEPS", "2000"))
    lr = float(os.environ.get("SOBOLERNA_POLISH_LR", "0.01"))
    return _load_jax_engine()["polisher"](raw_coords, contact_map, n_steps=n_steps, lr=lr)


def jax_energy(coords: np.ndarray, contact_map: np.ndarray | None) -> float:
    return _load_jax_engine()["energy_fn"](coords, contact_map)


def polish_candidate(
    raw_coords: np.ndarray,
    contact_map: np.ndarray | None,
    target_id: str = "",
    source: str = "",
    rank: int = 0,
    slot: int = 0,
    polisher: Callable[[np.ndarray, np.ndarray | None], np.ndarray] | None = None,
    energy_fn: Callable[[np.ndarray, np.ndarray | None], float] | None = None,
) -> PolishResult:
    raw = np.asarray(raw_coords, dtype=np.float64)
    metrics = PolishMetrics()
    fallback = PolishResult(
        target_id=target_id,
        source=source,
        rank=int(rank),
        slot=int(slot),
        accepted=False,
        reject_reason="not_run",
        metrics=metrics,
        raw_coords=raw,
        refined_coords=raw.copy(),
    )

    if not coords_are_valid(raw):
        fallback.reject_reason = "raw_invalid"
        return fallback

    if polisher is None:
        polisher = jax_polisher
    if energy_fn is None:
        energy_fn = jax_energy

    try:
        refined = np.asarray(polisher(raw, contact_map), dtype=np.float64)
    except Exception as exc:
        fallback.reject_reason = f"polish_failed:{type(exc).__name__}"
        return fallback

    result = PolishResult(
        target_id=target_id,
        source=source,
        rank=int(rank),
        slot=int(slot),
        accepted=False,
        reject_reason="rejected",
        metrics=metrics,
        raw_coords=raw,
        refined_coords=refined,
    )

    if refined.shape != raw.shape:
        result.reject_reason = "shape_mismatch"
        return result
    if not coords_are_valid(refined):
        result.reject_reason = "refined_invalid"
        return result

    metrics.bond_raw = bond_violation_count(raw)
    metrics.bond_refined = bond_violation_count(refined)
    if metrics.bond_refined > metrics.bond_raw:
        result.reject_reason = "bond_worsened"
        return result

    metrics.clash_raw = steric_clash_count(raw)
    metrics.clash_refined = steric_clash_count(refined)
    if metrics.clash_refined > metrics.clash_raw:
        result.reject_reason = "clash_worsened"
        return result

    try:
        metrics.energy_raw = float(energy_fn(raw, contact_map))
        metrics.energy_refined = float(energy_fn(refined, contact_map))
    except Exception as exc:
        result.reject_reason = f"energy_failed:{type(exc).__name__}"
        return result
    if not np.isfinite(metrics.energy_raw) or not np.isfinite(metrics.energy_refined):
        result.reject_reason = "energy_nonfinite"
        return result
    if metrics.energy_refined >= metrics.energy_raw:
        result.reject_reason = "energy_not_improved"
        return result

    metrics.rg_refined = radius_of_gyration(refined)
    metrics.expected_rg = expected_rg(len(raw))
    if not (0.7 * metrics.expected_rg <= metrics.rg_refined <= 1.5 * metrics.expected_rg):
        result.reject_reason = "rg_out_of_range"
        return result

    metrics.max_step_refined = max_consecutive_distance(refined)
    if metrics.max_step_refined >= 12.0:
        result.reject_reason = "max_step_too_large"
        return result

    metrics.tm_self = tm_self_score(raw, refined)
    if metrics.tm_self < 0.85:
        result.reject_reason = "tm_self_too_low"
        return result

    result.accepted = True
    result.reject_reason = ""
    return result


def _parse_slots_env() -> set[int] | None:
    raw = os.environ.get("SOBOLERNA_POLISH_SLOTS")
    if not raw:
        return None
    slots = set()
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        slots.add(int(piece))
    return slots


def _write_report(rows: list[dict], report_path: str | None, slots_path: str | None):
    if not report_path and not slots_path:
        return
    try:
        import pandas as pd

        report = pd.DataFrame(rows)
        if report_path:
            report.to_csv(report_path, index=False)
        if slots_path:
            cols = ["target_id", "source", "rank", "slot", "accepted", "reject_reason"]
            existing = [c for c in cols if c in report.columns]
            report[existing].to_csv(slots_path, index=False)
    except Exception as exc:
        print(f"[Sobolev polish] report write failed: {exc}")


def apply_guarded_sobolev_polish_to_submission(
    merged_pred,
    test_sequences,
    contact_provider: Callable | None = None,
    segments_map=None,
    context: Mapping | None = None,
    enabled: bool | None = None,
    report_path: str | None = SOBOLERNA_REPORT,
    slots_path: str | None = SOBOLERNA_SLOTS,
):
    if enabled is None:
        enabled = env_flag("SOBOLERNA_POLISH", True)
    if not enabled:
        print("[Sobolev polish] disabled; leaving 1st-place slots unchanged.")
        return merged_pred, []

    selected_slots = _parse_slots_env()
    output = merged_pred.copy()
    reports: list[dict] = []

    try:
        rows_iter = test_sequences.to_dict("records")
    except AttributeError:
        rows_iter = list(test_sequences)

    for row in rows_iter:
        target_id = row["target_id"]
        sequence = row["sequence"]
        seq_len = len(sequence)
        contact_map = build_contact_map(
            sequence=sequence,
            target_id=target_id,
            contact_provider=contact_provider,
            segments_map=segments_map,
            context=context,
        )
        plan = slot_plan_for_length(seq_len)
        for slot in range(1, 6):
            if selected_slots is not None and slot not in selected_slots:
                continue
            source, rank = plan[slot]
            try:
                raw_coords = coords_from_slot(output, target_id, slot)
                result = polish_candidate(
                    raw_coords=raw_coords,
                    contact_map=contact_map,
                    target_id=target_id,
                    source=source,
                    rank=rank,
                    slot=slot,
                )
            except Exception as exc:
                metrics = PolishMetrics()
                result = PolishResult(
                    target_id=target_id,
                    source=source,
                    rank=rank,
                    slot=slot,
                    accepted=False,
                    reject_reason=f"candidate_failed:{type(exc).__name__}",
                    metrics=metrics,
                    raw_coords=np.empty((0, 3)),
                    refined_coords=np.empty((0, 3)),
                )
            if result.accepted:
                output = write_coords_to_slot(
                    output, target_id, slot, result.refined_coords
                )
                print(
                    f"[Sobolev polish] accepted {target_id} slot {slot} "
                    f"({source}_{rank}) tm_self={result.metrics.tm_self:.3f}"
                )
            else:
                print(
                    f"[Sobolev polish] kept raw {target_id} slot {slot} "
                    f"({source}_{rank}): {result.reject_reason}"
                )
            reports.append(result.report_row())

    _write_report(reports, report_path, slots_path)
    accepted = sum(bool(r["accepted"]) for r in reports)
    print(f"[Sobolev polish] accepted {accepted}/{len(reports)} candidates.")
    return output, reports


def rows_for_polished_slot(final_submission_rows: Sequence[Mapping], slot: int) -> list[dict]:
    x_key, y_key, z_key = slot_coord_cols(slot)
    rows = []
    for row in final_submission_rows:
        rows.append(
            {
                "ID": row["ID"],
                "resname": row["resname"],
                "resid": row["resid"],
                "x": row.get(x_key, ""),
                "y": row.get(y_key, ""),
                "z": row.get(z_key, ""),
            }
        )
    return rows


def iter_accepted_polished_slots(slots_csv_path: str):
    import csv

    if not os.path.exists(slots_csv_path):
        return
    with open(slots_csv_path, newline="") as handle:
        for row in csv.DictReader(handle):
            accepted = str(row.get("accepted", "")).strip().lower()
            if accepted not in {"true", "1", "yes"}:
                continue
            try:
                slot = int(row["slot"])
                rank = int(row["rank"])
            except Exception:
                continue
            yield {
                "target_id": row["target_id"],
                "source": row.get("source", "sobolev"),
                "rank": rank,
                "slot": slot,
            }

