import numpy as np

from pythermalcomfort.classes_input import RidgeRegressionInputs
from pythermalcomfort.classes_return import PredictedBodyTemperatures
from pythermalcomfort.shared_functions import valid_range
from pythermalcomfort.utilities import Sex

# --- Model Constants ---

# Minimum values for each of the 8 input features, used for Min-Max scaling.
# The features are: Sex, Age, Height, Mass, Ambient Temp, Humidity, Rectal Temp (t_re), Mean Skin Temp (t_sk).
_FEATURES_SCALER_MIN = np.array(
    [
        0.0,
        -0.3114754098360656,
        -3.020833333333333,
        -0.6183587248751761,
        -1.222222222222222,
        -0.21951219512195122,
        -11.197107405358395,
        -2.105553500973682,
    ]
)

# Scaling factors (1 / (max - min)) for each of the 8 input features, used for Min-Max scaling.
_FEATURES_SCALER_SCALE = np.array(
    [
        1.0,
        0.01639344262295082,
        0.020833333333333332,
        0.012802458071949815,
        0.05555555555555555,
        0.024390243902439025,
        0.31613056192207334,
        0.08119094120192633,
    ]
)

# Minimum values for the 2 output variables (Rectal Temp, Mean Skin Temp), used for inverse scaling.
_OUTPUT_SCALER_MIN = np.array(
    [
        -11.197107405358395,
        -2.0197777680408033,
    ]
)

# Scaling factors (1 / (max - min)) for the 2 output variables, used for inverse scaling.
_OUTPUT_SCALER_SCALE = np.array(
    [
        0.31613056192207334,
        0.07894843838015173,
    ]
)

# Coefficients for the Rectal Temperature (t_re) ridge regression model.
# Corresponds to the 8 input features in order.
_T_RE_COEFFS = np.array(
    [
        0.00016261586852849347,
        0.0007368142143779594,
        -0.00043916987857211637,
        0.00046532701146677997,
        0.0008443934806620367,
        0.0006663379066237714,
        0.9932810428489056,
        0.006016233208250791,
    ]
)
# Intercept term for the Rectal Temperature (t_re) ridge regression model.
_T_RE_INTERCEPT = -0.0013528489525256315

# Coefficients for the Mean Skin Temperature (t_sk) ridge regression model.
# Corresponds to the 8 input features in order.
_T_SK_COEFFS = np.array(
    [
        0.0006157845452869151,
        0.00014854705372386215,
        -0.0004329826169348138,
        -0.0011471088118388912,
        0.018904677058503336,
        0.003188995712763656,
        -0.0010477636196332153,
        0.933918210580563,
    ]
)
# Intercept term for the Mean Skin Temperature (t_sk) ridge regression model.
_T_SK_INTERCEPT = 0.04356328728329839


def _scale_features(features: np.ndarray) -> np.ndarray:
    """Scales input features using predefined scaling constants."""
    return features * _FEATURES_SCALER_SCALE + _FEATURES_SCALER_MIN


def _inverse_scale_output(scaled_output: np.ndarray) -> np.ndarray:
    """Apply inverse scaling to the model output."""
    return (scaled_output - _OUTPUT_SCALER_MIN) / _OUTPUT_SCALER_SCALE


def _predict_temperature_simulation(features, duration):
    """Core simulation loop for temperature prediction."""
    # Scale input features
    scaled_features = _scale_features(features)

    # Precompute static components of the regression
    static_t_re = np.dot(scaled_features[:, :6], _T_RE_COEFFS[:6]) + _T_RE_INTERCEPT
    static_t_sk = np.dot(scaled_features[:, :6], _T_SK_COEFFS[:6]) + _T_SK_INTERCEPT

    # Initialize temperatures from scaled features
    prev_t_re = scaled_features[:, 6]
    prev_t_sk = scaled_features[:, 7]

    # Store history of temperatures
    t_re_history_scaled = []
    t_sk_history_scaled = []

    # Run simulation
    for _ in range(duration):
        new_t_re = (
            static_t_re + _T_RE_COEFFS[6] * prev_t_re + _T_RE_COEFFS[7] * prev_t_sk
        )
        new_t_sk = (
            static_t_sk + _T_SK_COEFFS[6] * prev_t_re + _T_SK_COEFFS[7] * prev_t_sk
        )
        prev_t_re, prev_t_sk = new_t_re, new_t_sk

        t_re_history_scaled.append(new_t_re)
        t_sk_history_scaled.append(new_t_sk)

    # Convert history lists to numpy arrays and transpose for easier processing
    # Shape becomes (n_scenarios, duration)
    t_re_history_scaled = np.array(t_re_history_scaled).T
    t_sk_history_scaled = np.array(t_sk_history_scaled).T

    # Flatten histories for scaling, then stack
    scaled_output = np.stack(
        [t_re_history_scaled.ravel(), t_sk_history_scaled.ravel()], axis=1
    )

    # Scale back the output
    final_temps_history_flat = _inverse_scale_output(scaled_output)

    # Reshape back to (n_scenarios, duration)
    n_scenarios = features.shape[0]
    t_re_history = final_temps_history_flat[:, 0].reshape(n_scenarios, duration)
    t_sk_history = final_temps_history_flat[:, 1].reshape(n_scenarios, duration)

    return t_re_history, t_sk_history


def _check_ridge_regression_compliance(age, height, weight, tdb, rh):
    """Check if the inputs are within the model's applicability limits."""
    age_valid = valid_range(age, (60, 100))
    height_valid = valid_range(height, (130, 230))
    mass_valid = valid_range(weight, (40, 140))
    temp_valid = valid_range(tdb, (0, 60))
    rh_valid = valid_range(rh, (0, 100))
    return age_valid, height_valid, mass_valid, temp_valid, rh_valid


def ridge_regression_body_temperature_predictor(
    sex: Sex | list[Sex],
    age: float | list[float],
    height: float | list[float],
    weight: float | list[float],
    tdb: float | list[float],
    rh: float | list[float],
    duration: int,
    initial_t_re: float | list[float] | None = None,
    initial_t_sk: float | list[float] | None = None,
    limit_inputs: bool = True,
    round_output: bool = True,
) -> PredictedBodyTemperatures:
    """Predicts the full history of core and skin temperature changes based on a ridge regression model.

    This model simulates the rectal (t_re) and mean skin (t_sk)
    temperatures by first establishing a baseline in a thermoneutral environment
    (23°C, 50% RH for 120 mins) and then simulating exposure to the specified
    environmental conditions for `duration`. If a `initial_t_re` and
    `initial_t_sk` are provided, then the initial 120 min simulation is skipped.
    Coefficients are taken from fold one from the study [Forbes2025]_
    (https://doi.org/10.1016/j.jtherbio.2025.104078). See notes documentation
    for limitations with this model.

    Parameters
    ----------
    sex : Sex or list
        Biological sex. Pass the string value from the enum, e.g.,
        `Sex.male.value` for "male" or `Sex.female.value` for "female".
    age : float or list
        Age, in years.
    height : float
        Body height [m].
    weight : float
        Body weight [kg].
    tdb : float or list
        Ambient (dry bulb) air temperature, in °C.
    rh : float or list
        Relative humidity, in %.
    duration : int, optional
        Duration of the simulation in the specified environment, in minutes.
    initial_t_re : float or list, optional
        Initial rectal temperature (°C). If provided, the baseline simulation
        at 23°C, 50% RH for 120 minutes is skipped, and this value is used
        as the starting rectal temperature for the main simulation.
    initial_t_sk : float or list, optional
        Initial mean skin temperature (°C). If provided, the baseline simulation
        at 23°C, 50% RH for 120 minutes is skipped, and this value is used
        as the starting mean skin temperature for the main simulation.
    limit_inputs : bool, optional
        If True, limits the inputs to the standard applicability limits. Defaults to True.
    round_output : bool, optional
        If True, rounds output value. If False, it does not round it. Defaults to True.

    Returns
    -------
    PredictedBodyTemperatures
        A dataclass containing the predicted rectal (`t_re`) and skin (`t_sk`)
        temperature history in °C. See
        :py:class:`~pythermalcomfort.classes_return.PredictedBodyTemperatures`
        for more details. The outputs are numpy arrays. For scalar inputs, the shape
        is (`duration`,). For vector inputs, the shape is (`n_inputs`, `duration`).

    Raises
    ------
    ValueError
        If input arrays have inconsistent shapes after broadcasting.

    Notes
    -----
    The model is applicable for adults aged between 60 and 100 years. The ranges for
    the input parameters are:

    - **Age**: 60 to 100 years
    - **Height**: 1.30 to 2.30 m
    - **Weight**: 40 to 140 kg
    - **Ambient Temperature**: 0 to 60 °C
    - **Relative Humidity**: 0 to 100 %

    The `limit_inputs` parameter, by default, is set to `True`, which means that
    if the inputs are outside the model's applicability limits, the function will
    return `nan`.

    The model does not have inputs such as: air velocity, radiative heat transfer,
    clothing level and an individual's activity level due to there being little
    variation in the dataset.

    All individuals in the dataset were in minimal clothing while stationary in a
    heat chamber, therefore stationary individuals in minimal clothing are best
    represented by this model.

    The function supports vectorized inputs for all parameters, allowing for
    batch predictions.

    Examples
    --------
    >>> from pythermalcomfort.utilities import Sex
    >>> from pythermalcomfort.data_driven.ridge_regression_body_temperature_predictor import (
    ...     ridge_regression_body_temperature_predictor,
    ... )
    >>> # Scalar example for a single person
    >>> results = ridge_regression_body_temperature_predictor(
    ...     sex=Sex.male.value,
    ...     age=60,
    ...     height=1.8,
    ...     weight=75,
    ...     tdb=35,
    ...     rh=60,
    ...     duration=540,
    ... )
    >>> print(f"Final Rectal temp: {results.t_re[-1]:.2f}°C")
    Final rectal temp: 37.98°C
    >>> print(f"Final Skin temp: {results.t_sk[-1]:.2f}°C")
    Final Skin temp: 37.02°C
    >>> print(f"Rectal temp history shape: {results.t_re.shape}")
    Rectal temp history shape: (540,)

    >>> # Vectorized example for multiple scenarios
    >>> results_vec = ridge_regression_body_temperature_predictor(
    ...     sex=[Sex.male.value, Sex.female.value],
    ...     age=[60, 65],
    ...     height=[1.8, 1.65],
    ...     weight=[75, 60],
    ...     tdb=[35, 40],
    ...     rh=[60, 50],
    ...     duration=540,
    ... )
    >>> print(f"Final rectal temps: {results_vec.t_re[:, -1]}")
    Final rectal temps: [37.98, 38.42]
    >>> print(f"Rectal temp history shape: {results_vec.t_re.shape}")
    Rectal temp history shape: (2, 540)
    """
    # Validate inputs
    RidgeRegressionInputs(
        sex=sex,
        age=age,
        height=height,
        weight=weight,
        tdb=tdb,
        rh=rh,
        duration=duration,
        initial_t_re=initial_t_re,
        initial_t_sk=initial_t_sk,
        limit_inputs=limit_inputs,
        round_output=round_output,
    )

    # Convert height from m to cm, handling both scalar and list-like inputs
    height_cm = np.asarray(height) * 100

    # Convert sex to 0 or 1 representation
    sex_array = np.array(sex)
    valid_sex_values = [Sex.male.value, Sex.female.value]
    if not np.all(np.isin(sex_array, valid_sex_values)):
        message = f"Invalid input for sex. Must be one of {valid_sex_values}"
        raise ValueError(message)

    # Vectorize sex input: 1 for female, 0 for male
    sex_value = np.where(sex_array == Sex.female.value, 1, 0)

    # Convert inputs to numpy arrays for vectorization
    inputs = np.broadcast_arrays(sex_value, age, height_cm, weight, tdb, rh)
    original_shape = inputs[0].shape
    flat_inputs = [np.ravel(i) for i in inputs]

    if limit_inputs:
        (
            age_valid,
            height_valid,
            mass_valid,
            temp_valid,
            rh_valid,
        ) = _check_ridge_regression_compliance(
            flat_inputs[1],
            flat_inputs[2],
            flat_inputs[3],
            flat_inputs[4],
            flat_inputs[5],
        )

    if initial_t_re is not None and initial_t_sk is not None:
        # Use provided baseline temperatures
        current_t_re = np.broadcast_to(initial_t_re, original_shape).ravel()
        current_t_sk = np.broadcast_to(initial_t_sk, original_shape).ravel()
    else:
        # Baseline simulation (120 mins in a neutral environment)
        initial_t_re = np.full_like(flat_inputs[0], 37.0, dtype=float)
        initial_t_sk = np.full_like(flat_inputs[0], 32.0, dtype=float)

        baseline_features = np.stack(
            [
                flat_inputs[0],  # sex
                flat_inputs[1],  # age
                flat_inputs[2],  # height
                flat_inputs[3],  # weight
                np.full_like(flat_inputs[0], 23.0),  # tdb = 23°C
                np.full_like(flat_inputs[0], 50.0),  # humidity = 50%
                initial_t_re,
                initial_t_sk,
            ],
            axis=1,
        )

        baseline_t_re_hist, baseline_t_sk_hist = _predict_temperature_simulation(
            baseline_features, duration=120
        )
        current_t_re = baseline_t_re_hist[:, -1]
        current_t_sk = baseline_t_sk_hist[:, -1]

    # Final simulation in specified environment
    final_features = np.stack(
        [
            flat_inputs[0],  # sex
            flat_inputs[1],  # age
            flat_inputs[2],  # height
            flat_inputs[3],  # weight
            flat_inputs[4],  # tdb
            flat_inputs[5],  # humidity
            current_t_re,
            current_t_sk,
        ],
        axis=1,
    )

    final_t_re_hist, final_t_sk_hist = _predict_temperature_simulation(
        final_features,
        duration=duration,
    )

    if limit_inputs:
        all_valid = ~(
            np.isnan(age_valid)
            | np.isnan(height_valid)
            | np.isnan(mass_valid)
            | np.isnan(temp_valid)
            | np.isnan(rh_valid)
        )
        # Apply validity mask across the time dimension
        validity_mask = all_valid[:, np.newaxis]
        final_t_re_hist = np.where(validity_mask, final_t_re_hist, np.nan)
        final_t_sk_hist = np.where(validity_mask, final_t_sk_hist, np.nan)

    if round_output:
        final_t_re_hist = np.round(final_t_re_hist, 2)
        final_t_sk_hist = np.round(final_t_sk_hist, 2)

    # If original input was scalar, return 1D array instead of 2D
    if not original_shape:
        final_t_re = final_t_re_hist.squeeze()
        final_t_sk = final_t_sk_hist.squeeze()
    else:
        final_t_re = final_t_re_hist
        final_t_sk = final_t_sk_hist

    return PredictedBodyTemperatures(t_re=final_t_re, t_sk=final_t_sk)
