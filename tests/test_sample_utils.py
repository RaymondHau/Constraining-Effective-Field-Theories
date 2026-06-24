from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np


source_path = Path(__file__).resolve().parents[1] / "scripts" / "EFT_prepare_samples.py"
tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
selected = [
    node for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    and node.name in {"draw_uniform_reference_events", "finite_target_mask"}
]
namespace = {"np": np}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)
draw_uniform_reference_events = namespace["draw_uniform_reference_events"]
finite_target_mask = namespace["finite_target_mask"]


class SampleUtilityTests(unittest.TestCase):
    def test_default_target_mask_does_not_truncate_finite_outliers(self) -> None:
        log_r = np.array([0.0, 50.0, np.nan])
        score = np.array([[0.0, 0.0], [20.0, -30.0], [0.0, 0.0]])
        self.assertEqual(finite_target_mask(log_r, score).tolist(), [True, True, False])

    def test_optional_target_limits_remain_available(self) -> None:
        log_r = np.array([0.0, 11.0])
        score = np.array([[0.0, 0.0], [6.0, 0.0]])
        self.assertEqual(finite_target_mask(log_r, score, 10.0, 5.0).tolist(), [True, False])

    def test_reference_events_are_drawn_uniformly_from_direct_reference_pool(self) -> None:
        candidates = np.arange(4)
        draws = draw_uniform_reference_events(candidates, 40_000, np.random.default_rng(7))
        fractions = np.bincount(draws, minlength=4) / len(draws)
        self.assertTrue(np.all(np.abs(fractions - 0.25) < 0.015))


if __name__ == "__main__":
    unittest.main()
