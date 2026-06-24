import json
from pathlib import Path

import numpy as np

from scripts.constrains import ScoreCalibration, ThetaPoint, calibrated_score_test, local_theta_grid, validation_datasets
from scripts.validation_events import portable_project_path


def test_stale_absolute_manifest_path_is_rebased(tmp_path: Path) -> None:
    output_dir = tmp_path / "madgraph_work" / "validation_events"
    theta_tag = "c1_p10_c2_p0"
    local_features = output_dir / theta_tag / "features.npy"
    local_features.parent.mkdir(parents=True)
    local_features.touch()

    validation_config = tmp_path / "validation.json"
    validation_config.write_text(json.dumps({"output_dir": "madgraph_work/validation_events"}))
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "theta_tag": theta_tag,
                        "theta_true": [10.0, 0.0],
                        "feature_file": "/home/someone/other_repo/madgraph_work/validation_events/c1_p10_c2_p0/features.npy",
                    }
                ]
            }
        )
    )

    config = {
        "_project_dir": str(tmp_path),
        "validation_config": str(validation_config),
        "datasets_from_validation": True,
    }
    datasets = validation_datasets(config)
    assert Path(datasets[0]["event_file"]) == local_features


def test_generated_metadata_path_is_project_relative(tmp_path: Path) -> None:
    feature_file = tmp_path / "madgraph_work" / "validation_events" / "features.npy"
    assert portable_project_path({"_project_dir": str(tmp_path)}, feature_file) == str(
        Path("madgraph_work") / "validation_events" / "features.npy"
    )


def test_calibrated_score_test_is_centered_and_covariance_normalized() -> None:
    calibration = ScoreCalibration(
        theta=np.array([[0.0, 0.0]]),
        mean=np.array([[0.1, -0.2]]),
        covariance=np.array([[[4.0, 0.0], [0.0, 9.0]]]),
        counts=np.array([100]),
        coordinate_scale=np.ones(2),
        neighbors=1,
        covariance_ridge_fraction=0.0,
    )
    event_count = 100
    expected_sum = event_count * calibration.mean[0]
    one_sigma_c1 = expected_sum + np.array([np.sqrt(event_count * 4.0), 0.0])
    score_sums = np.stack([expected_sum, one_sigma_c1])
    _, centered, q_score = calibrated_score_test(
        score_sums,
        [ThetaPoint(0.0, 0.0), ThetaPoint(0.0, 0.0)],
        event_count,
        calibration,
    )

    np.testing.assert_allclose(centered[0], 0.0)
    # The held-out mean is itself estimated from 100 events, so its uncertainty
    # doubles the total covariance when the constrained sample also has N=100.
    np.testing.assert_allclose(q_score, [0.0, 0.5])


def test_even_local_grid_contains_validation_truth_exactly() -> None:
    truth = ThetaPoint(0.0, 0.0)
    c1, c2, _ = local_theta_grid({}, (-1.0, 1.0, -2.0, 2.0), {"bins": 50}, truth)
    assert truth.c1 in c1
    assert truth.c2 in c2
