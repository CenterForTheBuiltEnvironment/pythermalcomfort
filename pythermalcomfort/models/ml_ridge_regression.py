from dataclasses import dataclass
from enum import Enum

import numpy as np

# --- Model Constants ---

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

_OUTPUT_SCALER_MIN = np.array(
    [
        -11.197107405358395,
        -2.0197777680408033,
    ]
)

_OUTPUT_SCALER_SCALE = np.array(
    [
        0.31613056192207334,
        0.07894843838015173,
    ]
)

_TRE_COEFFS = np.array(
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
_TRE_INTERCEPT = -0.0013528489525256315

_MTSK_COEFFS = np.array(
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
_MTSK_INTERCEPT = 0.04356328728329839


class Sex(Enum):
    MALE = 0
    FEMALE = 1


@dataclass
class PredictedTemperatures:
    """Dataclass for returning predicted temperature.

    Attributes
    ----------
    rectal_temp : float or numpy.ndarray
        Predicted rectal temperature (°C).
    skin_temp : float or numpy.ndarray
        Predicted mean skin temperature (°C).
    """

    rectal_temp: float | np.ndarray
    skin_temp: float | np.ndarray


def _scale_features(features: np.ndarray) -> np.ndarray:
    """Scales input features using predefined scaling constants."""
    return features * _FEATURES_SCALER_SCALE + _FEATURES_SCALER_MIN


def _inverse_scale_output(scaled_output: np.ndarray) -> np.ndarray:
    """Apply inverse scaling to the model output."""
    return (scaled_output - _OUTPUT_SCALER_MIN) / _OUTPUT_SCALER_SCALE


def _predict_temperature_simulation(features, duration_minutes):
    """Core simulation loop for temperature prediction."""
    # Scale input features
    scaled_features = _scale_features(features)

    # Precompute static components of the regression
    static_tre = np.dot(scaled_features[:, :6], _TRE_COEFFS[:6]) + _TRE_INTERCEPT
    static_mtsk = np.dot(scaled_features[:, :6], _MTSK_COEFFS[:6]) + _MTSK_INTERCEPT

    # Initialize temperatures from scaled features
    prev_tre = scaled_features[:, 6]
    prev_mtsk = scaled_features[:, 7]

    # Run simulation
    for _ in range(duration_minutes):
        new_tre = static_tre + _TRE_COEFFS[6] * prev_tre + _TRE_COEFFS[7] * prev_mtsk
        new_mtsk = (
            static_mtsk + _MTSK_COEFFS[6] * prev_tre + _MTSK_COEFFS[7] * prev_mtsk
        )
        prev_tre, prev_mtsk = new_tre, new_mtsk

    # Scale back the final output and return
    scaled_output = np.stack([prev_tre, prev_mtsk], axis=1)
    final_temps = _inverse_scale_output(scaled_output)
    return final_temps[:, 0], final_temps[:, 1]


def ridge_regression_predictor(
    sex: int | list | np.ndarray,
    age: float | list | np.ndarray,
    height_cm: float | list | np.ndarray,
    mass_kg: float | list | np.ndarray,
    ambient_temp: float | list | np.ndarray,
    humidity: float | list | np.ndarray,
    duration_minutes: int,
    baseline_tre: float | list | np.ndarray | None = None,
    baseline_mtsk: float | list | np.ndarray | None = None,
) -> PredictedTemperatures:
    """Predicts core and skin temperature changes based on a ridge regression model.

    This model simulates the rectal (tre) and mean skin (mtsk)
    temperatures by first establishing a baseline in a thermoneutral environment
    (23°C, 50% RH for 120 mins) and then simulating exposure to the specified
    environmental conditions for `duration_minutes`. If a `baseline_tre` and
    `baseline_mtsk` are provided, then the initial 120 min simulation is skipped.
    Coefficients are taken from fold one from the study [Forbes2025]_
    (https://doi.org/10.1016/j.jtherbio.2025.104078). See notes documentation
    for limitations with this model.

    Parameters
    ----------
    sex : int, list, or numpy.ndarray
        Biological sex, where 0 for Male, 1 for Female.
    age : float, list, or numpy.ndarray
        Age, in years.
    height_cm : float, list, or numpy.ndarray
        Height, in centimeters.
    mass_kg : float, list, or numpy.ndarray
        Body mass, in kilograms.
    ambient_temp : float, list, or numpy.ndarray
        Ambient air temperature, in °C.
    humidity : float, list, or numpy.ndarray
        Relative humidity, in %.
    duration_minutes : int, optional
        Duration of the simulation in the specified environment, in minutes.
    baseline_tre : float, list, or numpy.ndarray, optional
        Initial rectal temperature (°C). If provided, the baseline simulation
        at 23°C, 50% RH for 120 minutes is skipped, and this value is used
        as the starting rectal temperature for the main simulation.
    baseline_mtsk : float, list, or numpy.ndarray, optional
        Initial mean skin temperature (°C). If provided, the baseline simulation
        at 23°C, 50% RH for 120 minutes is skipped, and this value is used
        as the starting mean skin temperature for the main simulation.

    Returns
    -------
    PredictedTemperatures
        A dataclass containing the predicted rectal (`rectal_temp`)
        and skin (`skin_temp`) temperatures in °C.

    Raises
    ------
    ValueError
        If input arrays have inconsistent shapes after broadcasting.

    Notes
    -----
    This model was trained on adults over 60 years therefore may not give accurate
    predictions for people under 60 years old.

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
    >>> from pythermalcomfort.models.ml_ridge_regression import (
    ...     ridge_regression_predictor,
    ...     Sex,
    ... )
    >>> # Scalar example for a single person
    >>> results = ridge_regression_predictor(
    ...     sex=Sex.MALE.value,
    ...     age=60,
    ...     height_cm=180,
    ...     mass_kg=75,
    ...     ambient_temp=35,
    ...     humidity=60,
    ...     duration_minutes=540,
    ... )
    >>> print(f"Rectal temp: {results.rectal_temp:.2f}°C")
    Rectal temp: 37.98°C
    >>> print(f"Skin temp: {results.skin_temp:.2f}°C")
    Skin temp: 37.02°C

    >>> # Vectorized example for multiple scenarios
    >>> results_vec = ridge_regression_predictor(
    ...     sex=[Sex.MALE.value, Sex.FEMALE.value],
    ...     age=[60, 65],
    ...     height_cm=[180, 165],
    ...     mass_kg=[75, 60],
    ...     ambient_temp=[35, 40],
    ...     humidity=[60, 50],
    ...     duration_minutes=540,
    ... )
    >>> print(results_vec.rectal_temp)
    [37.98... 38.42...]

    >>> # Example with provided baseline temperatures
    >>> results_baseline = ridge_regression_predictor(
    ...     sex=Sex.MALE.value,
    ...     age=70,
    ...     height_cm=180,
    ...     mass_kg=75,
    ...     ambient_temp=35,
    ...     humidity=60,
    ...     duration_minutes=60,
    ...     baseline_tre=37.0,
    ...     baseline_mtsk=32.0,
    ... )
    >>> print(f"Rectal temp: {results_baseline.rectal_temp:.2f}°C")
    Rectal temp (from baseline): 37.33°C
    """
    # Convert inputs to numpy arrays for vectorization
    inputs = np.broadcast_arrays(sex, age, height_cm, mass_kg, ambient_temp, humidity)
    original_shape = inputs[0].shape
    flat_inputs = [np.ravel(i) for i in inputs]

    if baseline_tre is not None and baseline_mtsk is not None:
        # Use provided baseline temperatures
        current_tre = np.broadcast_to(baseline_tre, original_shape).ravel()
        current_mtsk = np.broadcast_to(baseline_mtsk, original_shape).ravel()
    else:
        # Baseline simulation (120 mins in a neutral environment)
        initial_tre = np.full_like(flat_inputs[0], 37.0, dtype=float)
        initial_mtsk = np.full_like(flat_inputs[0], 32.0, dtype=float)

        baseline_features = np.stack(
            [
                flat_inputs[0],  # sex
                flat_inputs[1],  # age
                flat_inputs[2],  # height_cm
                flat_inputs[3],  # mass_kg
                np.full_like(flat_inputs[0], 23.0),  # ambient_temp = 23°C
                np.full_like(flat_inputs[0], 50.0),  # humidity = 50%
                initial_tre,
                initial_mtsk,
            ],
            axis=1,
        )

        current_tre, current_mtsk = _predict_temperature_simulation(
            baseline_features, duration_minutes=120
        )

    # Final simulation in specified environment
    final_features = np.stack(
        [
            flat_inputs[0],  # sex
            flat_inputs[1],  # age
            flat_inputs[2],  # height_cm
            flat_inputs[3],  # mass_kg
            flat_inputs[4],  # ambient_temp
            flat_inputs[5],  # humidity
            current_tre,
            current_mtsk,
        ],
        axis=1,
    )

    final_tre, final_mtsk = _predict_temperature_simulation(
        final_features,
        duration_minutes=duration_minutes,
    )

    # Reshape results to match original input shape
    if original_shape:
        final_tre = final_tre.reshape(original_shape)
        final_mtsk = final_mtsk.reshape(original_shape)
    else:  # Handle scalar case
        final_tre = final_tre.item()
        final_mtsk = final_mtsk.item()

    return PredictedTemperatures(rectal_temp=final_tre, skin_temp=final_mtsk)
