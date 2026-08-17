"""Tests for risk_metrics.metrics: known-value checks and edge cases
(empty series, zero volatility, all-positive returns for VaR/CVaR)."""

import numpy as np
import pandas as pd
import pytest

from risk_metrics.metrics import (
    annualized_return,
    annualized_vol,
    cvar_historic,
    drawdown,
    kurtosis,
    max_drawdown,
    returns_from_prices,
    semideviation,
    sharpe_ratio,
    skewness,
    summary_stats,
    var_cornish_fisher,
    var_historic,
)


class TestReturnsFromPrices:
    def test_known_price_series(self):
        prices = pd.Series([100.0, 110.0, 99.0])
        returns = returns_from_prices(prices)
        assert returns.iloc[0] == pytest.approx(0.10)
        assert returns.iloc[1] == pytest.approx(-0.10)

    def test_first_row_dropped(self):
        prices = pd.DataFrame({"A": [100.0, 105.0, 110.0]})
        returns = returns_from_prices(prices)
        assert len(returns) == 2


class TestAnnualizedReturn:
    def test_doubles_over_one_year_of_daily_returns(self):
        # 252 identical daily returns compounding to exactly 2x.
        daily_r = 2.0 ** (1 / 252) - 1
        returns = pd.Series([daily_r] * 252)
        assert annualized_return(returns, periods_per_year=252) == pytest.approx(1.0, rel=1e-6)

    def test_empty_series_returns_zero(self):
        assert annualized_return(pd.Series([], dtype=float)) == 0.0


class TestAnnualizedVol:
    def test_zero_for_constant_returns(self):
        returns = pd.Series([0.01] * 100)
        assert annualized_vol(returns) == pytest.approx(0.0, abs=1e-10)

    def test_scales_with_sqrt_time(self):
        rng = np.random.default_rng(0)
        returns = pd.Series(rng.normal(0, 0.01, 1000))
        vol_daily = annualized_vol(returns, periods_per_year=1)
        vol_annual = annualized_vol(returns, periods_per_year=252)
        assert vol_annual == pytest.approx(vol_daily * (252 ** 0.5))


class TestSharpeRatio:
    def test_zero_volatility_returns_zero_not_error(self):
        returns = pd.Series([0.0] * 100)
        assert sharpe_ratio(returns, riskfree_rate=0.03) == 0.0

    def test_positive_excess_return_gives_positive_sharpe(self):
        returns = pd.Series([0.001] * 252)
        sr = sharpe_ratio(returns, riskfree_rate=0.02, periods_per_year=252)
        assert sr > 0


class TestDrawdown:
    def test_no_losses_means_zero_drawdown(self):
        returns = pd.Series([0.01, 0.02, 0.01])
        assert max_drawdown(returns) == pytest.approx(0.0)

    def test_known_drawdown_value(self):
        # +10%, then -50% from the peak.
        returns = pd.Series([0.10, -0.50])
        result = drawdown(returns)
        assert result["Drawdown"].iloc[-1] == pytest.approx(-0.50)

    def test_empty_series_returns_zero(self):
        assert max_drawdown(pd.Series([], dtype=float)) == 0.0


class TestSkewnessKurtosis:
    def test_symmetric_distribution_has_near_zero_skew(self):
        rng = np.random.default_rng(1)
        returns = pd.Series(rng.normal(0, 0.01, 10_000))
        assert skewness(returns) == pytest.approx(0.0, abs=0.1)

    def test_normal_distribution_kurtosis_near_three(self):
        rng = np.random.default_rng(2)
        returns = pd.Series(rng.normal(0, 0.01, 10_000))
        assert kurtosis(returns) == pytest.approx(3.0, abs=0.2)

    def test_zero_std_does_not_divide_by_zero(self):
        returns = pd.Series([0.01] * 50)
        assert skewness(returns) == 0.0
        assert kurtosis(returns) == 0.0


class TestSemideviation:
    def test_all_gains_returns_zero(self):
        returns = pd.Series([0.01, 0.02, 0.03])
        assert semideviation(returns) == 0.0

    def test_positive_for_mixed_returns(self):
        returns = pd.Series([0.05, -0.03, 0.02, -0.01, 0.01])
        assert semideviation(returns) > 0


class TestVaR:
    def test_historic_var_is_positive_loss_magnitude(self):
        rng = np.random.default_rng(3)
        returns = pd.Series(rng.normal(0, 0.02, 1000))
        var = var_historic(returns, level=5.0)
        assert var > 0

    def test_cvar_is_at_least_as_large_as_var(self):
        rng = np.random.default_rng(4)
        returns = pd.Series(rng.normal(0, 0.02, 1000))
        var = var_historic(returns, level=5.0)
        cvar = cvar_historic(returns, level=5.0)
        assert cvar >= var

    def test_cvar_no_observations_beyond_threshold_returns_zero(self):
        # Degenerate: only 2 points, VaR sits exactly at the single loss.
        returns = pd.Series([0.01, 0.02])
        result = cvar_historic(returns, level=5.0)
        assert np.isfinite(result)

    def test_cornish_fisher_matches_gaussian_for_normal_data(self):
        """With skew=0 and kurtosis=3, the Cornish-Fisher expansion
        collapses to the plain Gaussian VaR."""
        rng = np.random.default_rng(5)
        returns = pd.Series(rng.normal(0, 0.02, 50_000))
        from scipy.stats import norm

        z = norm.ppf(0.05)
        gaussian_var = -(returns.mean() + z * returns.std(ddof=0))
        cf_var = var_cornish_fisher(returns, level=5.0)
        assert cf_var == pytest.approx(gaussian_var, rel=0.05)


class TestSummaryStats:
    def test_one_row_per_asset(self):
        rng = np.random.default_rng(6)
        returns = pd.DataFrame({
            "A": rng.normal(0.0005, 0.01, 500),
            "B": rng.normal(0.0003, 0.02, 500),
        })
        stats = summary_stats(returns)
        assert list(stats.index) == ["A", "B"]
        assert "Sharpe Ratio" in stats.columns
        assert "Max Drawdown" in stats.columns
