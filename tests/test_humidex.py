from pythermalcomfort.models import humidex


def test_humidex():
    assert humidex(25, 50) == {"humidex": 28.2, "discomfort": "Little or no discomfort"}
    assert humidex(30, 80) == {
        "humidex": 43.3,
        "discomfort": "Intense discomfort; avoid exertion",
    }
    assert humidex(31.6, 57.1) == {
        "humidex": 40.8,
        "discomfort": "Intense discomfort; avoid exertion",
    }
