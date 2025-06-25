import pytest
import numpy as np
import pandas as pd
from definition_ed9255a35d8048d090966abb43f6db33 import calculate_idiosyncratic_returns

@pytest.mark.parametrize("returns, factor_estimates, params, expected_type", [
    # Basic numeric numpy arrays, params typical
    (np.array([0.01, 0.02, 0.015, 0.03]), np.array([0.008, 0.018, 0.02, 0.025]), {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1}, np.ndarray),
    # Basic pandas Series input
    (pd.Series([0.01, 0.02, 0.015, 0.03]), pd.Series([0.008, 0.018, 0.02, 0.025]), {"n_estimators": 100, "max_depth": 2, "learning_rate": 0.05}, pd.Series),
    # Empty input arrays
    (np.array([]), np.array([]), {"n_estimators": 10, "max_depth": 1, "learning_rate": 0.1}, ValueError),
    # Mismatched lengths - returns longer than factor_estimates
    (np.array([0.01, 0.02, 0.015]), np.array([0.008, 0.018]), {"n_estimators": 10, "max_depth": 1, "learning_rate": 0.1}, ValueError),
    # Mismatched lengths - factor_estimates longer than returns
    (np.array([0.01, 0.02]), np.array([0.008, 0.018, 0.02]), {"n_estimators": 10, "max_depth": 1, "learning_rate": 0.1}, ValueError),
    # params missing required keys
    (np.array([0.01, 0.02, 0.015]), np.array([0.008, 0.018, 0.02]), {}, ValueError),
    # params with invalid types for hyperparameters
    (np.array([0.01, 0.02, 0.015]), np.array([0.008, 0.018, 0.02]), {"n_estimators": "fifty", "max_depth": 3, "learning_rate": 0.1}, TypeError),
    (np.array([0.01, 0.02, 0.015]), np.array([0.008, 0.018, 0.02]), {"n_estimators": 10, "max_depth": -1, "learning_rate": 0.1}, ValueError),
    # returns with NaN values
    (np.array([0.01, np.nan, 0.015, 0.02]), np.array([0.008, 0.018, 0.02, 0.02]), {"n_estimators": 10, "max_depth": 1, "learning_rate": 0.1}, ValueError),
    # factor_estimates with NaN values
    (np.array([0.01, 0.02, 0.015, 0.02]), np.array([0.008, np.nan, 0.02, 0.02]), {"n_estimators": 10, "max_depth": 1, "learning_rate": 0.1}, ValueError),
    # inputs are lists (should be accepted or produce TypeError)
    ([0.01, 0.02, 0.015, 0.02], [0.008, 0.018, 0.02, 0.02], {"n_estimators": 10, "max_depth": 1, "learning_rate": 0.1}, (np.ndarray, pd.Series)),
    # Very large arrays - perform type check only (not testing performance)
    (np.random.random(1000), np.random.random(1000), {"n_estimators": 20, "max_depth": 2, "learning_rate": 0.1}, np.ndarray),
    # params with extra keys - should ignore or accept
    (np.array([0.01, 0.02, 0.015]), np.array([0.008, 0.018, 0.02]), {"n_estimators": 10, "max_depth": 1, "learning_rate": 0.1, "extra_param": 123}, np.ndarray),
    # params with zero n_estimators
    (np.array([0.01, 0.02, 0.015]), np.array([0.008, 0.018, 0.02]), {"n_estimators": 0, "max_depth": 1, "learning_rate": 0.1}, ValueError),
    # params with learning_rate zero or negative
    (np.array([0.01, 0.02, 0.015]), np.array([0.008, 0.018, 0.02]), {"n_estimators": 10, "max_depth": 1, "learning_rate": 0}, ValueError),
    (np.array([0.01, 0.02, 0.015]), np.array([0.008, 0.018, 0.02]), {"n_estimators": 10, "max_depth": 1, "learning_rate": -0.1}, ValueError),

    # None inputs for arrays
    (None, np.array([0.008, 0.018, 0.02]), {"n_estimators": 10, "max_depth": 1, "learning_rate": 0.1}, TypeError),
    (np.array([0.01, 0.02, 0.015]), None, {"n_estimators": 10, "max_depth": 1, "learning_rate": 0.1}, TypeError),

])
def test_calculate_idiosyncratic_returns(returns, factor_estimates, params, expected_type):
    if isinstance(expected_type, tuple):
        # Accept multiple possible types
        try:
            result = calculate_idiosyncratic_returns(returns, factor_estimates, params)
            assert isinstance(result, expected_type)
        except Exception as e:
            assert any(isinstance(e, exc) for exc in expected_type)
    elif issubclass(expected_type, Exception):
        with pytest.raises(expected_type):
            calculate_idiosyncratic_returns(returns, factor_estimates, params)
    else:
        result = calculate_idiosyncratic_returns(returns, factor_estimates, params)
        assert isinstance(result, expected_type)
        # Length of output matches inputs
        # Accepting pandas Series or ndarray
        assert len(result) == len(returns)
