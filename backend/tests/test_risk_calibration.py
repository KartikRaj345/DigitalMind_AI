import numpy as np

from backend.app.risk.calibration import (
    Calibrator,
    brier_score,
    expected_calibration_error,
    reliability_table,
)


def test_isotonic_fixes_overconfidence():
    # Raw scores are wildly over-confident: 0.95 predicted, 50% correct.
    raw = np.array([0.95] * 20 + [0.05] * 20)
    truth = np.array([1.0] * 10 + [0.0] * 10 + [1.0] * 3 + [0.0] * 17)
    cal = Calibrator.fit(raw, truth, method="isotonic")
    out = cal.apply(raw[:1])
    assert 0.4 < float(out[0]) < 0.6


def test_isotonic_is_monotone():
    raw = np.linspace(0, 1, 50)
    truth = (raw > 0.5).astype(float)
    cal = Calibrator.fit(raw, truth, method="isotonic")
    applied = cal.apply(raw)
    assert np.all(np.diff(applied) >= -1e-9)


def test_platt_recovers_linear_map():
    raw = np.linspace(0.01, 0.99, 400)
    z = np.log(raw / (1 - raw))
    truth = 1.0 / (1.0 + np.exp(-(0.5 * z - 1.0)))
    truth = (np.random.default_rng(0).random(400) < truth).astype(float)
    cal = Calibrator.fit(raw, truth, method="platt")
    assert abs(cal.slope - 0.5) < 0.25
    assert abs(cal.intercept + 1.0) < 0.35


def test_ece_is_zero_when_empirical_rate_equals_confidence():
    # Each bin must have enough samples that its empirical *rate* (not a single 0/1
    # outcome) can equal its mean confidence. We use 100 samples per level.
    conf, truth = [], []
    for level, n_pos in ((0.05, 5), (0.25, 25), (0.55, 55), (0.85, 85)):
        conf.extend([level] * 100)
        truth.extend([1.0] * n_pos + [0.0] * (100 - n_pos))
    assert expected_calibration_error(np.array(conf), np.array(truth), n_bins=10) < 1e-9


def test_ece_is_large_for_overconfident_predictions():
    assert expected_calibration_error(np.array([0.9, 0.9]), np.array([0.0, 1.0]), n_bins=10) > 0.3


def test_brier():
    assert brier_score(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 0.0
    assert brier_score(np.array([0.0, 1.0]), np.array([1.0, 0.0])) == 1.0


def test_roundtrip_dict():
    raw = np.linspace(0.01, 0.99, 120)
    truth = (raw > 0.4).astype(float)
    cal = Calibrator.fit(raw, truth)
    restored = Calibrator.from_dict(cal.to_dict())
    assert np.allclose(cal.apply(raw), restored.apply(raw))


def test_empty_input_safe():
    cal = Calibrator.fit(np.array([]), np.array([]))
    assert float(cal.apply(np.array([0.5]))[0]) == 0.5


def test_reliability_table_shape():
    raw = np.random.default_rng(1).random(100)
    truth = (np.random.default_rng(2).random(100) < raw).astype(float)
    rows = reliability_table(raw, truth, n_bins=10)
    assert len(rows) == 10
    assert sum(r["count"] for r in rows) == 100
