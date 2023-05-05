import pytest
from pythermalcomfort.jos3_functions.thermoregulation import conv_coef

def test_conv_coef():
    # Default test
    result = conv_coef(posture="standing", v=0.1, tdb=28.8, t_skin=34.0)
    print("Default test results", result)

    # Test valuables
    postures = ["standing", "sitting", "lying", "sedentary", "supine"]
    velocities = [0.1, 0.3]
    air_temps = [28.8, 30.0]
    skin_temps = [34.0, 35.0]

    for posture in postures:
        for velocity in velocities:
            for air_temp in air_temps:
                for skin_temp in skin_temps:
                    result = conv_coef(posture=posture, v=velocity, tdb=air_temp, t_skin=skin_temp)
                    print(f"Results（posture={posture}, v={velocity}, tdb={air_temp}, t_skin={skin_temp}）: {result}")


test_conv_coef()
