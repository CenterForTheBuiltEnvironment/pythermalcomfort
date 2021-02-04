import warnings
import math

warnings.simplefilter("always")


def transpose_sharp_altitude(sharp, altitude):
    altitude_new = math.degrees(math.asin(
        math.sin(math.radians(abs(sharp - 90))) * math.cos(math.radians(altitude))))
    sharp = math.degrees(math.atan(
        math.sin(math.radians(sharp)) * math.tan(math.radians(90 - altitude))))
    sol_altitude = altitude_new
    return round(sharp,3), round(sol_altitude,3)


def check_standard_compliance(standard, **kwargs):
    params = dict()
    params["standard"] = standard
    for key, value in kwargs.items():
        params[key] = value

    if params["standard"] == "utci":
        for key, value in params.items():
            if key == "v" and (value > 17 or value < 0.5):
                warnings.warn(
                    "UTCI wind speed applicability limits between 0.5 and 17 m/s",
                    UserWarning,
                )

    if params["standard"] == "ankle_draft":
        for key, value in params.items():
            if key == "met" and value > 1.3:
                warnings.warn(
                    "The ankle draft model is only valid for met <= 1.3", UserWarning,
                )
            if key == "clo" and value > 0.7:
                warnings.warn(
                    "The ankle draft model is only valid for clo <= 0.7", UserWarning,
                )

    elif params["standard"] == "ashrae":  # based on table 7.3.4 ashrae 55 2017
        for key, value in params.items():
            if key in ["tdb", "tr"]:
                if key == "tdb":
                    parameter = "dry-bulb"
                else:
                    parameter = "mean radiant"
                if value > 40 or value < 10:
                    warnings.warn(
                        f"ASHRAE {parameter} temperature applicability limits between 10 and 40 °C",
                        UserWarning,
                    )
            if key in ["v", "vr"] and (value > 2 or value < 0):
                warnings.warn(
                    "ASHRAE air velocity applicability limits between 0 and 2 m/s",
                    UserWarning,
                )
            if key == "met" and (value > 2 or value < 1):
                warnings.warn(
                    "ASHRAE met applicability limits between 1.0 and 2.0 met",
                    UserWarning,
                )
            if key == "clo" and (value > 1.5 or value < 0):
                warnings.warn(
                    "ASHRAE clo applicability limits between 0.0 and 1.5 clo",
                    UserWarning,
                )
            if key == "v_limited" and value > 0.2:
                raise ValueError(
                    "This equation is only applicable for air speed lower than 0.2 m/s"
                )

    elif params["standard"] == "iso":  # based on ISO 7730:2005 page 3
        for key, value in params.items():
            if key == "tdb" and (value > 30 or value < 10):
                warnings.warn(
                    "ISO air temperature applicability limits between 10 and 30 °C",
                    UserWarning,
                )
            if key == "tr" and (value > 40 or value < 10):
                warnings.warn(
                    "ISO mean radiant temperature applicability limits between 10 and 40 °C",
                    UserWarning,
                )
            if key in ["v", "vr"] and (value > 1 or value < 0):
                warnings.warn(
                    "ISO air velocity applicability limits between 0 and 1 m/s",
                    UserWarning,
                )
            if key == "met" and (value > 4 or value < 0.8):
                warnings.warn(
                    "ISO met applicability limits between 0.8 and 4.0 met", UserWarning,
                )
            if key == "clo" and (value > 2 or value < 0):
                warnings.warn(
                    "ISO clo applicability limits between 0.0 and 2 clo", UserWarning,
                )


#: This dictionary contains the met values of typical tasks.
met_typical_tasks = {
    "Sleeping": 0.7,
    "Reclining": 0.8,
    "Seated, quiet": 1.0,
    "Reading, seated": 1.0,
    "Writing": 1.0,
    "Typing": 1.1,
    "Standing, relaxed": 1.2,
    "Filing, seated": 1.2,
    "Flying aircraft, routine": 1.2,
    "Filing, standing": 1.4,
    "Driving a car": 1.5,
    "Walking about": 1.7,
    "Cooking": 1.8,
    "Table sawing": 1.8,
    "Walking 2mph (3.2kmh)": 2.0,
    "Lifting/packing": 2.1,
    "Seated, heavy limb movement": 2.2,
    "Light machine work": 2.2,
    "Flying aircraft, combat": 2.4,
    "Walking 3mph (4.8kmh)": 2.6,
    "House cleaning": 2.7,
    "Driving, heavy vehicle": 3.2,
    "Dancing": 3.4,
    "Calisthenics": 3.5,
    "Walking 4mph (6.4kmh)": 3.8,
    "Tennis": 3.8,
    "Heavy machine work": 4.0,
    "Handling 100lb (45 kg) bags": 4.0,
    "Pick and shovel work": 4.4,
    "Basketball": 6.3,
    "Wrestling": 7.8,
}

#: This dictionary contains the total clothing insulation of typical typical ensembles.
clo_typical_ensembles = {
    "Walking shorts, short-sleeve shirt": 0.36,
    "Typical summer indoor clothing": 0.5,
    "Knee-length skirt, short-sleeve shirt, sandals, underwear": 0.54,
    "Trousers, short-sleeve shirt, socks, shoes, underwear": 0.57,
    "Trousers, long-sleeve shirt": 0.61,
    "Knee-length skirt, long-sleeve shirt, full slip": 0.67,
    "Sweat pants, long-sleeve sweatshirt": 0.74,
    "Jacket, Trousers, long-sleeve shirt": 0.96,
    "Typical winter indoor clothing": 1.0,
}

#: This dictionary contains the clo values of individual clothing elements. To calculate the total clothing insulation you need to add these values together.
clo_individual_garments = {
    "Metal chair": 0.00,
    "Bra": 0.01,
    "Wooden stool": 0.01,
    "Ankle socks": 0.02,
    "Shoes or sandals": 0.02,
    "Slippers": 0.03,
    "Panty hose": 0.02,
    "Calf length socks": 0.03,
    "Women's underwear": 0.03,
    "Men's underwear": 0.04,
    "Knee socks (thick)": 0.06,
    "Short shorts": 0.06,
    "Walking shorts": 0.08,
    "T-shirt": 0.08,
    "Standard office chair": 0.10,
    "Executive chair": 0.15,
    "Boots": 0.1,
    "Sleeveless scoop-neck blouse": 0.12,
    "Half slip": 0.14,
    "Long underwear bottoms": 0.15,
    "Full slip": 0.16,
    "Short-sleeve knit shirt": 0.17,
    "Sleeveless vest (thin)": 0.1,
    "Sleeveless vest (thick)": 0.17,
    "Sleeveless short gown (thin)": 0.18,
    "Short-sleeve dress shirt": 0.19,
    "Sleeveless long gown (thin)": 0.2,
    "Long underwear top": 0.2,
    "Thick skirt": 0.23,
    "Long-sleeve dress shirt": 0.25,
    "Long-sleeve flannel shirt": 0.34,
    "Long-sleeve sweat shirt": 0.34,
    "Short-sleeve hospital gown": 0.31,
    "Short-sleeve short robe (thin)": 0.34,
    "Short-sleeve pajamas": 0.42,
    "Long-sleeve long gown": 0.46,
    "Long-sleeve short wrap robe (thick)": 0.48,
    "Long-sleeve pajamas (thick)": 0.57,
    "Long-sleeve long wrap robe (thick)": 0.69,
    "Thin trousers": 0.15,
    "Thick trousers": 0.24,
    "Sweatpants": 0.28,
    "Overalls": 0.30,
    "Coveralls": 0.49,
    "Thin skirt": 0.14,
    "Long-sleeve shirt dress (thin)": 0.33,
    "Long-sleeve shirt dress (thick)": 0.47,
    "Short-sleeve shirt dress": 0.29,
    "Sleeveless, scoop-neck shirt (thin)": 0.23,
    "Sleeveless, scoop-neck shirt (thick)": 0.27,
    "Long sleeve shirt (thin)": 0.25,
    "Long sleeve shirt (thick)": 0.36,
    "Single-breasted coat (thin)": 0.36,
    "Single-breasted coat (thick)": 0.44,
    "Double-breasted coat (thin)": 0.42,
    "Double-breasted coat (thick)": 0.48,
}
