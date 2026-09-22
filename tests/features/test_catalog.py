"""The catalogue's value is what it refuses, so each refusal is pinned here."""

from __future__ import annotations

import numpy as np
import pytest

from core import audit
from core.features.catalog import (
    Feature,
    FeatureCatalog,
    Rejection,
    catalogue_summary,
    rank_correlation,
)


def price_panel(n_rows: int = 120, n_cols: int = 5, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0003, 0.011, size=(n_rows, n_cols))
    return 100.0 * np.exp(np.cumsum(steps, axis=0))


def momentum_20(close: np.ndarray) -> np.ndarray:
    out = np.zeros_like(close)  # explicit warm-up, never NaN
    if close.shape[0] > 20:
        out[20:] = close[20:] / close[:-20] - 1.0
    return out


def momentum_20_in_log(close: np.ndarray) -> np.ndarray:
    """A monotone transform of momentum_20. Pearson would let this in; rank will not."""
    return np.log1p(np.clip(momentum_20(close), -0.99, None))


def realised_vol_20(close: np.ndarray) -> np.ndarray:
    out = np.zeros_like(close)
    returns = np.zeros_like(close)
    returns[1:] = close[1:] / close[:-1] - 1.0
    for t in range(21, close.shape[0]):
        out[t] = returns[t - 20 : t + 1].std(axis=0)
    return out


def tomorrows_return(close: np.ndarray) -> np.ndarray:
    """The feature every backtest wishes it had. Registration must refuse it."""
    out = np.zeros_like(close)
    out[:-1] = close[1:] / close[:-1] - 1.0
    return out


def nan_warmup(close: np.ndarray) -> np.ndarray:
    out = np.full_like(close, np.nan)
    out[20:] = close[20:] / close[:-20] - 1.0
    return out


def wrong_shape(close: np.ndarray) -> np.ndarray:
    return close.mean(axis=1)


def feature(name: str, fn, owner: str = "alpha-pod-equity-statarb") -> Feature:
    return Feature(name=name, description=f"{name} test feature", source="close panel", owner=owner, fn=fn)


def test_a_clean_feature_is_accepted():
    catalog = FeatureCatalog(price_panel())
    result = catalog.register(feature("momentum_20", momentum_20))
    assert result.accepted
    assert result.rejection is None
    assert result.leak_probes > 0
    assert catalog.names() == ("momentum_20",)


def test_a_feature_that_reads_the_future_is_refused_at_registration():
    """G0 would also catch it, but by then every alpha built on it is waste."""
    catalog = FeatureCatalog(price_panel())
    result = catalog.register(feature("tomorrows_return", tomorrows_return))
    assert not result.accepted
    assert result.rejection is Rejection.LEAKAGE
    assert "look-ahead" in result.reason
    assert catalog.names() == ()


def test_a_monotone_restatement_is_a_duplicate():
    catalog = FeatureCatalog(price_panel())
    assert catalog.register(feature("momentum_20", momentum_20)).accepted

    result = catalog.register(feature("momentum_20_log", momentum_20_in_log))
    assert not result.accepted
    assert result.rejection is Rejection.DUPLICATE
    assert "momentum_20" in result.reason
    assert abs(result.correlations["momentum_20"]) > 0.9


def test_an_independent_feature_is_not_a_duplicate():
    catalog = FeatureCatalog(price_panel())
    assert catalog.register(feature("momentum_20", momentum_20)).accepted
    result = catalog.register(feature("realised_vol_20", realised_vol_20))
    assert result.accepted, result.reason
    assert abs(result.correlations["momentum_20"]) <= 0.90
    assert set(catalog.names()) == {"momentum_20", "realised_vol_20"}


def test_nan_warmup_is_refused_with_an_instruction():
    catalog = FeatureCatalog(price_panel())
    result = catalog.register(feature("nan_warmup", nan_warmup))
    assert not result.accepted
    assert result.rejection is Rejection.NON_FINITE
    assert "warm-up" in result.reason


def test_wrong_shape_is_refused():
    catalog = FeatureCatalog(price_panel())
    result = catalog.register(feature("wrong_shape", wrong_shape))
    assert not result.accepted
    assert result.rejection is Rejection.SHAPE


def test_a_name_may_not_be_reused():
    catalog = FeatureCatalog(price_panel())
    assert catalog.register(feature("momentum_20", momentum_20)).accepted
    result = catalog.register(feature("momentum_20", realised_vol_20))
    assert not result.accepted
    assert result.rejection is Rejection.NAME_TAKEN
    # The first registration still stands, unchanged.
    assert np.allclose(catalog.values("momentum_20"), momentum_20(catalog.panel))


def test_every_pod_reads_every_feature():
    """CLAUDE.md rule 6: there is no private feature, so there is no owner filter."""
    catalog = FeatureCatalog(price_panel())
    catalog.register(feature("momentum_20", momentum_20, owner="alpha-pod-trend-macro"))
    values = catalog.values("momentum_20")
    assert values.shape == price_panel().shape
    assert catalog.feature("momentum_20").owner == "alpha-pod-trend-macro"


def test_unknown_feature_raises_rather_than_returning_zeros():
    catalog = FeatureCatalog(price_panel())
    with pytest.raises(KeyError, match="no feature named"):
        catalog.values("does_not_exist")


def test_registrations_and_refusals_are_both_audited(tmp_path):
    log = tmp_path / "audit.log"
    catalog = FeatureCatalog(price_panel(), audit_path=log)
    catalog.register(feature("momentum_20", momentum_20))
    catalog.register(feature("tomorrows_return", tomorrows_return))

    records = list(audit.read(log))
    assert [r["event"] for r in records] == ["features.register", "features.register"]
    assert records[0]["data"]["accepted"] is True
    assert records[1]["data"]["accepted"] is False
    assert records[1]["data"]["rejection"] == "leakage"
    ok, broken = audit.verify(log)
    assert ok and broken is None


def test_pairwise_correlations_are_reported_for_the_crowding_review():
    catalog = FeatureCatalog(price_panel())
    catalog.register(feature("momentum_20", momentum_20))
    catalog.register(feature("realised_vol_20", realised_vol_20))
    pairs = catalog.correlations()
    assert set(pairs) == {("momentum_20", "realised_vol_20")}
    assert -1.0 <= next(iter(pairs.values())) <= 1.0


def test_summary_cites_the_catalogue_rather_than_recalling_it():
    catalog = FeatureCatalog(price_panel())
    catalog.register(feature("momentum_20", momentum_20, owner="feature-factory"))
    rows = catalogue_summary(catalog)
    assert rows == [
        {
            "name": "momentum_20",
            "owner": "feature-factory",
            "source": "close panel",
            "description": "momentum_20 test feature",
        }
    ]


def test_rank_correlation_of_a_constant_is_zero_not_nan():
    panel = price_panel()
    assert rank_correlation(panel, np.ones_like(panel)) == 0.0


def test_threshold_is_the_catalogue_rule_and_can_be_tightened_per_catalogue():
    """The number lives in the catalogue, not in the risk limit table (see ADR-0005)."""
    catalog = FeatureCatalog(price_panel(), max_abs_correlation=0.10)
    assert catalog.register(feature("momentum_20", momentum_20)).accepted
    result = catalog.register(feature("realised_vol_20", realised_vol_20))
    assert not result.accepted
    assert result.rejection is Rejection.DUPLICATE
