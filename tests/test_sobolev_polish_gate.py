import math
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import sobolev_polish_gate as gate


def make_compact_coords(n=80):
    radius = 12.0
    dz = 1.5
    theta = 2.0 * math.asin(math.sqrt(gate.IDEAL_C1_DISTANCE**2 - dz**2) / (2.0 * radius))
    i = np.arange(n, dtype=np.float64)
    return np.column_stack(
        [
            radius * np.cos(theta * i),
            radius * np.sin(theta * i),
            dz * (i - i.mean()),
        ]
    )


def low_energy(coords, _contact_map):
    return float(np.mean(np.sum(np.asarray(coords) ** 2, axis=1)))


def constant_energy(_coords, _contact_map):
    return 1.0


class SobolevPolishGateTests(unittest.TestCase):
    def test_slot_extract_write_roundtrip(self):
        coords = make_compact_coords(5)
        df = pd.DataFrame(
            {
                "ID": [f"T1_{i + 1}" for i in range(5)],
                "resname": list("ACGUA"),
                "resid": np.arange(1, 6),
                "x_1": coords[:, 0],
                "y_1": coords[:, 1],
                "z_1": coords[:, 2],
            }
        )
        got = gate.coords_from_slot(df, "T1", 1)
        np.testing.assert_allclose(got, coords)
        updated = coords + 1.0
        gate.write_coords_to_slot(df, "T1", 1, updated)
        np.testing.assert_allclose(gate.coords_from_slot(df, "T1", 1), updated)

    def test_disabled_apply_preserves_submission_exactly(self):
        coords = make_compact_coords(3)
        df = pd.DataFrame(
            {
                "ID": [f"T1_{i + 1}" for i in range(3)],
                "resname": list("ACG"),
                "resid": np.arange(1, 4),
                **{
                    f"{axis}_{slot}": coords[:, axis_idx]
                    for slot in range(1, 6)
                    for axis_idx, axis in enumerate(["x", "y", "z"])
                },
            }
        )
        test_sequences = pd.DataFrame({"target_id": ["T1"], "sequence": ["ACG"]})
        out, report = gate.apply_guarded_sobolev_polish_to_submission(
            df, test_sequences, enabled=False, report_path=None, slots_path=None
        )
        pd.testing.assert_frame_equal(out, df)
        self.assertEqual(report, [])

    def test_accepts_safe_energy_improving_polish(self):
        raw = make_compact_coords()
        refined = raw * 0.99
        result = gate.polish_candidate(
            raw,
            np.zeros((len(raw), len(raw))),
            polisher=lambda _raw, _cmap: refined,
            energy_fn=low_energy,
        )
        self.assertTrue(result.accepted, result.reject_reason)
        self.assertGreaterEqual(result.metrics.tm_self, 0.85)

    def test_rejects_shape_mismatch(self):
        raw = make_compact_coords()
        result = gate.polish_candidate(
            raw,
            None,
            polisher=lambda _raw, _cmap: raw[:-1],
            energy_fn=low_energy,
        )
        self.assertEqual(result.reject_reason, "shape_mismatch")

    def test_rejects_invalid_refined_coords(self):
        raw = make_compact_coords()
        refined = raw.copy()
        refined[0, 0] = np.nan
        result = gate.polish_candidate(
            raw,
            None,
            polisher=lambda _raw, _cmap: refined,
            energy_fn=low_energy,
        )
        self.assertEqual(result.reject_reason, "refined_invalid")

    def test_rejects_bond_worsening(self):
        raw = make_compact_coords()
        refined = raw.copy()
        refined[1] += np.array([20.0, 0.0, 0.0])
        result = gate.polish_candidate(
            raw,
            None,
            polisher=lambda _raw, _cmap: refined,
            energy_fn=low_energy,
        )
        self.assertEqual(result.reject_reason, "bond_worsened")

    def test_rejects_clash_worsening(self):
        raw = make_compact_coords()
        refined = raw.copy()
        refined[2] = refined[0]
        with mock.patch.object(gate, "bond_violation_count", side_effect=[0, 0]):
            result = gate.polish_candidate(
                raw,
                None,
                polisher=lambda _raw, _cmap: refined,
                energy_fn=low_energy,
            )
        self.assertEqual(result.reject_reason, "clash_worsened")

    def test_rejects_energy_not_improved(self):
        raw = make_compact_coords()
        refined = raw * 0.99
        result = gate.polish_candidate(
            raw,
            None,
            polisher=lambda _raw, _cmap: refined,
            energy_fn=constant_energy,
        )
        self.assertEqual(result.reject_reason, "energy_not_improved")

    def test_rejects_rg_out_of_range(self):
        raw = make_compact_coords()
        refined = raw * 0.2
        with mock.patch.object(gate, "bond_violation_count", side_effect=[0, 0]), \
             mock.patch.object(gate, "steric_clash_count", side_effect=[0, 0]):
            result = gate.polish_candidate(
                raw,
                None,
                polisher=lambda _raw, _cmap: refined,
                energy_fn=low_energy,
            )
        self.assertEqual(result.reject_reason, "rg_out_of_range")

    def test_rejects_max_step_too_large(self):
        raw = make_compact_coords()
        refined = raw * 0.99
        refined[20:] += np.array([30.0, 0.0, 0.0])
        with mock.patch.object(gate, "bond_violation_count", side_effect=[0, 0]), \
             mock.patch.object(gate, "steric_clash_count", side_effect=[0, 0]), \
             mock.patch.object(gate, "radius_of_gyration", return_value=gate.expected_rg(len(raw))):
            result = gate.polish_candidate(
                raw,
                None,
                polisher=lambda _raw, _cmap: refined,
                energy_fn=lambda coords, _cmap: 0.0 if coords is refined else 1.0,
            )
        self.assertEqual(result.reject_reason, "max_step_too_large")

    def test_rejects_tm_self_too_low(self):
        raw = make_compact_coords()
        refined = raw.copy()
        refined[::2] *= -1.0
        with mock.patch.object(gate, "bond_violation_count", side_effect=[0, 0]), \
             mock.patch.object(gate, "steric_clash_count", side_effect=[0, 0]), \
             mock.patch.object(gate, "radius_of_gyration", return_value=gate.expected_rg(len(raw))), \
             mock.patch.object(gate, "max_consecutive_distance", return_value=gate.IDEAL_C1_DISTANCE):
            result = gate.polish_candidate(
                raw,
                None,
                polisher=lambda _raw, _cmap: refined,
                energy_fn=lambda coords, _cmap: 0.0 if coords is refined else 1.0,
            )
        self.assertEqual(result.reject_reason, "tm_self_too_low")


if __name__ == "__main__":
    unittest.main()
