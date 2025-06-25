import pytest
import pandas as pd
import numpy as np
from definition_81ea799d652d4da087a49a0a689e5e2e import train_sdf_model

@pytest.fixture
def valid_data():
    # Example valid data with risk free rate and two public factors over 10 time points
    dates = pd.date_range("2020-01-01", periods=10)
    data = pd.DataFrame({
        "RiskFreeRate": np.linspace(0.01, 0.02, 10),
        "Factor1": np.linspace(1.0, 2.0, 10),
        "Factor2": np.linspace(0.5, 1.5, 10)
    }, index=dates)
    return data

@pytest.fixture
def initial_params_example():
    # Example initial params as a dictionary or any iterable structure expected
    return {"alpha": 0.01, "beta_1": 0.5, "beta_2": 0.3}

@pytest.mark.parametrize("data,initial_params,expected_exception", [
    # Valid case - dictionary/series-like initial params and DataFrame input
    (pytest.lazy_fixture('valid_data'), {"alpha": 0.0, "beta_1": 0.1, "beta_2": 0.2}, None),
    (pytest.lazy_fixture('valid_data'), {"alpha": 0.05, "beta_1": -0.1, "beta_2": 0.0}, None),

    # Edge cases with empty data
    (pd.DataFrame(), {"alpha": 0.0, "beta_1": 0.1, "beta_2": 0.2}, ValueError),

    # Data missing required columns
    (pd.DataFrame({"SomeColumn": [1,2,3]}), {"alpha": 0.01, "beta_1": 0.1}, ValueError),

    # Data with NaN values
    (pd.DataFrame({"RiskFreeRate": [0.01, np.nan], "Factor1": [1.0, 2.0], "Factor2": [0.5, 0.6]}), 
     {"alpha": 0.01, "beta_1": 0.1, "beta_2": 0.2}, ValueError),

    # initial_params not a dict or wrong type
    (pytest.lazy_fixture('valid_data'), None, TypeError),
    (pytest.lazy_fixture('valid_data'), [0.1, 0.2], TypeError),
    (pytest.lazy_fixture('valid_data'), "invalid_string", TypeError),

    # Data input not a DataFrame
    ([], {"alpha": 0.0, "beta_1": 0.1, "beta_2": 0.2}, TypeError),
    (None, {"alpha": 0.0, "beta_1": 0.1, "beta_2": 0.2}, TypeError),

    # Parameters with missing keys or wrong keys
    (pytest.lazy_fixture('valid_data'), {"beta_1": 0.1, "beta_2": 0.2}, ValueError),
    (pytest.lazy_fixture('valid_data'), {}, ValueError),

    # Data with non-numeric values or strings in numeric fields
    (pd.DataFrame({
        "RiskFreeRate": ["a", "b", "c"],
        "Factor1": [1, 2, 3],
        "Factor2": [0.1, 0.2, 0.3]
    }), {"alpha": 0.01, "beta_1": 0.1, "beta_2": 0.2}, ValueError),

    # Very large dataset edge case (performance) - here just sanity check
    (pd.DataFrame({
        "RiskFreeRate": np.ones(10000) * 0.01,
        "Factor1": np.ones(10000) * 1.0,
        "Factor2": np.ones(10000) * 0.5
    }), {"alpha": 0.01, "beta_1": 0.5, "beta_2": 0.3}, None),

])
def test_train_sdf_model(data, initial_params, expected_exception):
    if expected_exception is None:
        result = train_sdf_model(data, initial_params)
        # Result should be dict or DataFrame as per spec
        assert isinstance(result, (dict, pd.DataFrame))
        # Check if alpha present in result
        if isinstance(result, dict):
            assert "alpha" in result
        else:
            assert "alpha" in result.columns or "alpha" in result.index or "alpha" in result.keys()
    else:
        with pytest.raises(expected_exception):
            train_sdf_model(data, initial_params)