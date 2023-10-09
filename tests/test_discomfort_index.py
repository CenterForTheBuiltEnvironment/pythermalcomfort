import numpy as np

from pythermalcomfort.models import discomfort_index


def test_discomfort_index():
    np.testing.assert_equal(
        discomfort_index([21, 23.5, 29, 32, 35, 40], 50),
        {
            "di": [19.2, 21.0, 25.0, 27.2, 29.4, 33.0],
            "discomfort_condition": [
                "No discomfort",
                "Less than 50% feels discomfort",
                "More than 50% feels discomfort",
                "Most of the population feels discomfort",
                "Everyone feels severe stress",
                "State of medical emergency",
            ],
        },
    )
    np.testing.assert_equal(
        discomfort_index([35, 35], [10, 90]),
        {
            "di": [24.9, 33.9],
            "discomfort_condition": [
                "More than 50% feels discomfort",
                "State of medical emergency",
            ],
        },
    )
