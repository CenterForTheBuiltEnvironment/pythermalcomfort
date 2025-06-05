import numpy as np
import pytest

from pythermalcomfort.models.wind_chill_temperature import wct


class TestWct:
    """Test cases for the wind chill temperature (WCT) model."""

    # TODO this tests are not synced with comf R

    def test_wct_results(self) -> None:
        """Test that the function calculates WCT correctly for various inputs."""
        np.equal(wct(tdb=-20, v=5).wct, -24.3)
        np.equal(wct(tdb=-20, v=15).wct, -29.1)
        np.equal(wct(tdb=-20, v=60).wct, -36.5)

    # Calculate WCT correctly for single float inputs of temperature and wind speed
    def test_wct_single_float_inputs(self) -> None:
        """Test that the function calculates WCT correctly for single float values of tdb and v."""
        # Test with single float values
        result = wct(tdb=-5.0, v=5.5)

        # Expected value calculated using the formula:
        # 13.12 + 0.6215 * tdb - 11.37 * v**0.16 + 0.3965 * tdb * v**0.16
        expected = -7.5

        assert isinstance(result.wct, float)
        assert round(result.wct, 1) == expected

    # Handle empty lists for temperature and wind speed inputs
    def test_wct_empty_lists(self) -> None:
        """Test that the function handles empty lists for tdb and v inputs."""
        # Test with empty lists
        np.allclose(wct(tdb=[], v=[]).wct, np.array([]), equal_nan=True)

    # Calculate WCT correctly for lists of temperature and wind speed values
    def test_wct_list_inputs(self) -> None:
        """Test that the function calculates WCT correctly for lists of tdb and v values."""
        # Test with list of values
        result = wct(tdb=[-5.0, -10.0], v=[5.5, 10.0], round_output=True)

        # Expected values calculated using the formula:
        # 13.12 + 0.6215 * tdb - 11.37 * v**0.16 + 0.3965 * tdb * v**0.16
        expected = [-7.5, -15.3]

        assert isinstance(result.wct, np.ndarray)
        assert np.allclose(result.wct, expected, atol=0.1)

    # Handle non-numeric input values
    def test_wct_non_numeric_inputs(self) -> None:
        """Function raises a TypeError if non-numeric values are inputs."""
        # Test with non-numeric values for tdb and v
        with pytest.raises(TypeError):
            wct(tdb="invalid", v=5.5)

        with pytest.raises(TypeError):
            wct(tdb=-5.0, v="invalid")

        with pytest.raises(TypeError):
            wct(tdb="invalid", v="invalid")
