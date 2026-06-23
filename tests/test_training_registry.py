"""Unit tests for the algorithm trainer registry.

Fast checks that exercise the registry/resolution wiring without training any
models -- they guard the integration points each new algorithm must hook into.
"""

from __future__ import annotations

import pytest

from rl_stock.training import _CONTINUOUS_ONLY, _TRAINERS, _resolve_algorithms


def test_ddpg_is_registered_with_a_trainer() -> None:
    assert "DDPG" in _TRAINERS
    assert callable(_TRAINERS["DDPG"])


def test_ddpg_is_marked_continuous_only() -> None:
    assert "DDPG" in _CONTINUOUS_ONLY


def test_resolve_keeps_ddpg_for_position_env() -> None:
    assert _resolve_algorithms("position", ("DDPG",)) == ["DDPG"]


def test_resolve_skips_ddpg_for_discrete_t1_env() -> None:
    # Continuous-action algorithms are filtered out of the discrete t1 env.
    assert _resolve_algorithms("t1", ("DDPG",)) == []


def test_resolve_is_case_insensitive() -> None:
    assert _resolve_algorithms("position", ("ddpg",)) == ["DDPG"]


def test_resolve_preserves_order_alongside_other_algos() -> None:
    requested = ("ppo", "ddpg", "sac")
    assert _resolve_algorithms("position", requested) == ["PPO", "DDPG", "SAC"]


def test_resolve_rejects_unknown_algorithm() -> None:
    with pytest.raises(ValueError, match="Unknown algorithm"):
        _resolve_algorithms("position", ("XYZ",))
