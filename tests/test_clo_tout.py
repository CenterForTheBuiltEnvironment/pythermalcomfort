import unittest
import numpy as np
from pythermalcomfort.models import clo_tout


class TestCloTout(unittest.TestCase):
    def test_si_units_scalar(self):
        # Test SI units with a scalar outdoor air temperature
        result = clo_tout(tout=27)
        self.assertEqual(result, 0.46)

    def test_si_units_array(self):
        # Test SI units with an array of outdoor air temperatures
        result = clo_tout(tout=[27, 25])
        np.testing.assert_equal(result, [0.46, 0.47])

    def test_ip_units_scalar(self):
        # Test IP units with a scalar outdoor air temperature
        result = clo_tout(tout=80.6, units="IP")
        self.assertEqual(result, 0.46)

    def test_ip_units_array(self):
        # Test IP units with an array of outdoor air temperatures
        result = clo_tout(tout=[80.6, 77], units="IP")
        np.testing.assert_equal(result, np.array([0.46, 0.47]))

    def test_edge_cases(self):
        # Test edge cases of the piecewise function
        result = clo_tout(tout=[4, 2, 0])
        np.testing.assert_equal(result, [0.67, 0.75, 0.82])

        result = clo_tout(tout=[-4, -6])
        np.testing.assert_equal(result, [0.96, 1.0])

    def test_invalid_units(self):
        # Test invalid units
        with self.assertRaises(ValueError):
            clo_tout(tout=27, units="invalid")

    def test_invalid_type(self):
        # Test invalid type for tout
        with self.assertRaises(TypeError):
            clo_tout(tout="invalid")
