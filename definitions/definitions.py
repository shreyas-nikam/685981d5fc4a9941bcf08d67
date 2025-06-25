
from typing import Dict, Union
import pandas as pd
import numpy as np
from scipy.optimize import minimize

def train_sdf_model(data: pd.DataFrame, initial_params: Dict[str, float]) -> Union[Dict[str, float], pd.DataFrame]:
    """
    Trains the Stochastic Discount Factor (SDF) model using public market index data,
    estimating alpha and beta coefficients by fitting a multi-period return model.

    Args:
        data: DataFrame with columns 'RiskFreeRate', 'Factor1', 'Factor2'.
        initial_params: dict with keys 'alpha', 'beta_1', 'beta_2' (floats) as initial guesses.

    Returns:
        Dictionary with keys 'alpha', 'beta_1', 'beta_2' for estimated parameters.

    Raises:
        TypeError: If data is not a DataFrame or initial_params not a dict.
        ValueError: If data is empty, missing required columns, contains NaNs,
                    or initial_params keys missing or non-numeric.
        RuntimeError: If optimization fails.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data.empty:
        raise ValueError("data is empty")

    required_cols = {"RiskFreeRate", "Factor1", "Factor2"}
    if not required_cols.issubset(data.columns):
        raise ValueError(f"data must contain columns: {required_cols}")
    subset = data[list(required_cols)]
    if subset.isnull().values.any():
        raise ValueError("data contains NaN values in required columns")
    if not all(pd.api.types.is_numeric_dtype(subset[col]) for col in required_cols):
        raise ValueError("data columns must be numeric")

    if not isinstance(initial_params, dict):
        raise TypeError("initial_params must be a dictionary")
    required_keys = {"alpha", "beta_1", "beta_2"}
    if not required_keys.issubset(initial_params):
        raise ValueError(f"initial_params must contain keys: {required_keys}")
    if not all(isinstance(initial_params[k], (int, float)) for k in required_keys):
        raise ValueError("initial_params values must be numeric")

    # Define excess returns: factors minus risk free rate
    factors = data[["Factor1", "Factor2"]].values
    rf = data["RiskFreeRate"].values.reshape(-1, 1)
    excess_factors = factors - rf

    # SDF model: m_t * R_t+1 = 1
    # Assume m_t = alpha + beta1 * factor1 + beta2 * factor2
    # Objective: minimize sum of squared differences of 1 - m_t * (1 + factors)

    def objective(params: np.ndarray) -> float:
        alpha, beta1, beta2 = params
        m_t = alpha + beta1 * data["Factor1"].values + beta2 * data["Factor2"].values
        # Model-implied price: m_t * (1 + total return), here: factors + 1 (approximate total returns)
        residuals = 1 - m_t * (1 + factors.sum(axis=1))
        return np.sum(residuals ** 2)

    x0 = np.array([initial_params["alpha"], initial_params["beta_1"], initial_params["beta_2"]])
    result = minimize(objective, x0, method="BFGS")
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    alpha_opt, beta1_opt, beta2_opt = result.x
    return {"alpha": alpha_opt, "beta_1": beta1_opt, "beta_2": beta2_opt}


import numpy as np
import pandas as pd
from typing import Union, Dict
from sklearn.ensemble import GradientBoostingRegressor

def calculate_idiosyncratic_returns(
    returns: Union[np.ndarray, pd.Series, list],
    factor_estimates: Union[np.ndarray, pd.Series, list],
    params: Dict[str, object]
) -> Union[np.ndarray, pd.Series]:
    """
    Calculates idiosyncratic returns by modeling residuals with Componentwise L2 Boosting
    via sklearn's GradientBoostingRegressor.

    Args:
        returns: Actual return values (1D array-like).
        factor_estimates: Predicted returns from public factor model (1D array-like).
        params: Hyperparameters dict with keys 'n_estimators' (int > 0), 
                'max_depth' (int > 0), 'learning_rate' (float > 0).

    Returns:
        Estimated idiosyncratic returns, matching the type of inputs (np.ndarray or pd.Series).

    Raises:
        TypeError: If invalid types for inputs or params.
        ValueError: If inputs invalid length, contain NaNs, or params invalid values.
    """
    if returns is None or factor_estimates is None:
        raise TypeError("returns and factor_estimates cannot be None")

    # Determine input type for output
    returns_is_series = isinstance(returns, pd.Series)
    factor_is_series = isinstance(factor_estimates, pd.Series)

    # Convert to numpy arrays for processing
    try:
        ret_arr = np.asarray(returns)
        fac_arr = np.asarray(factor_estimates)
    except Exception:
        raise TypeError("returns and factor_estimates must be array-like")

    if ret_arr.ndim != 1 or fac_arr.ndim != 1:
        raise ValueError("returns and factor_estimates must be one-dimensional")

    if len(ret_arr) == 0 or len(fac_arr) == 0:
        raise ValueError("Input arrays must not be empty")

    if len(ret_arr) != len(fac_arr):
        raise ValueError("returns and factor_estimates must have the same length")

    if np.isnan(ret_arr).any():
        raise ValueError("returns contains NaN values")
    if np.isnan(fac_arr).any():
        raise ValueError("factor_estimates contains NaN values")

    # Validate params presence and types
    required = {"n_estimators", "max_depth", "learning_rate"}
    if not required.issubset(params):
        missing = required - params.keys()
        raise ValueError(f"Missing required parameters: {missing}")

    n_estimators = params["n_estimators"]
    max_depth = params["max_depth"]
    learning_rate = params["learning_rate"]

    if not isinstance(n_estimators, int):
        raise TypeError("n_estimators must be an int")
    if not isinstance(max_depth, int):
        raise TypeError("max_depth must be an int")
    if not isinstance(learning_rate, (float, int)):
        raise TypeError("learning_rate must be a float")

    if n_estimators <= 0:
        raise ValueError("n_estimators must be positive")
    if max_depth <= 0:
        raise ValueError("max_depth must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    residuals = ret_arr - fac_arr
    X = fac_arr.reshape(-1, 1)

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=0,
    )
    model.fit(X, residuals)
    pred_residuals = model.predict(X)

    if returns_is_series and factor_is_series:
        return pd.Series(pred_residuals, index=returns.index)
    return np.array(pred_residuals)
