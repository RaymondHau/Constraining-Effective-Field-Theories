from __future__ import annotations

import ast
import unittest
from pathlib import Path

import torch


source_path = Path(__file__).resolve().parents[1] / "scripts" / "EFT_train_estimators.py"
tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
selected = [
    node for node in tree.body
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    and node.name in {"numerator_score_loss", "rascal_ratio_loss"}
]
namespace = {"torch": torch}
exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"), namespace)
numerator_score_loss = namespace["numerator_score_loss"]
rascal_ratio_loss = namespace["rascal_ratio_loss"]


class TrainingLossTests(unittest.TestCase):
    def test_score_loss_uses_numerator_rows_only(self) -> None:
        prediction = torch.tensor([[100.0, -100.0], [2.0, 4.0]])
        target = torch.tensor([[0.0, 0.0], [1.0, 2.0]])
        y = torch.tensor([[0.0], [1.0]])

        # The denominator's huge residual is ignored. Eq. (35)/(37) averages the
        # numerator squared norm over the full two-row batch: (1^2 + 2^2) / 2.
        self.assertTrue(torch.isclose(numerator_score_loss(prediction, target, y), torch.tensor(2.5)))

    def test_rascal_uses_ratio_on_denominator_and_inverse_on_numerator(self) -> None:
        ratio = torch.tensor([[2.0], [4.0]])
        y = torch.tensor([[0.0], [1.0]])
        exact_log_ratio = torch.log(ratio)

        self.assertTrue(torch.isclose(rascal_ratio_loss(exact_log_ratio, ratio, y), torch.tensor(0.0)))

    def test_joint_log_regression_is_not_the_rascal_objective(self) -> None:
        ratio = torch.tensor([[2.0], [4.0]])
        y = torch.tensor([[0.0], [1.0]])
        wrong_log_ratio = torch.log(torch.tensor([[3.0], [3.0]]))

        expected = ((3.0 - 2.0) ** 2 + (1.0 / 3.0 - 1.0 / 4.0) ** 2) / 2.0
        self.assertTrue(torch.isclose(rascal_ratio_loss(wrong_log_ratio, ratio, y), torch.tensor(expected)))

if __name__ == "__main__":
    unittest.main()
