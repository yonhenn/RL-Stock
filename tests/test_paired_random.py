from __future__ import annotations

import numpy as np
import pytest

from scripts.evaluate_paired_random import empirical_percentile, paired_actions


@pytest.mark.unit
def test_paired_actions_share_uniform_draws():
    long_actions, symmetric_actions = paired_actions(seed=42, length=100)
    np.testing.assert_allclose(symmetric_actions, 2.0 * long_actions - 1.0)
    assert np.all((long_actions >= 0.0) & (long_actions <= 1.0))
    assert np.all((symmetric_actions >= -1.0) & (symmetric_actions <= 1.0))


@pytest.mark.unit
def test_paired_actions_are_reproducible_and_seeded():
    first = paired_actions(seed=42, length=10)
    repeated = paired_actions(seed=42, length=10)
    different = paired_actions(seed=43, length=10)
    np.testing.assert_array_equal(first[0], repeated[0])
    assert not np.array_equal(first[0], different[0])


@pytest.mark.unit
def test_empirical_percentile_higher_is_better():
    values = np.array([1.0, 2.0, 3.0])
    assert empirical_percentile(values, 2.0) == pytest.approx(60.0)


@pytest.mark.unit
def test_empirical_percentile_lower_is_better():
    values = np.array([0.1, 0.2, 0.3])
    assert empirical_percentile(values, 0.1, lower_is_better=True) == pytest.approx(80.0)
