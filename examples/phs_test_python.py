from phs7933_2023_model import PHS79332023

# Define test cases (5 working conditions)
cases = [
    # Case 1
    {
        "accl": 1, "posture": 1, "Ta": 40, "Tg": 40, "Va": 0.30, "RH": 35, "M": 300,
        "Icl": 0.5, "Ap": 0.0, "Fr": 0.0
    },
    # Case 2
    {
        "accl": 2, "posture": 1, "Ta": 35, "Tg": 35, "Va": 0.10, "RH": 60, "M": 300,
        "Icl": 0.5, "Ap": 0.0, "Fr": 0.0
    },
    # Case 3
    {
        "accl": 2, "posture": 1, "Ta": 30, "Tg": 45, "Va": 0.10, "RH": 35, "M": 300,
        "Icl": 0.8, "Ap": 0.30, "Fr": 0.85
    },
    # Case 4
    {
        "accl": 2, "posture": 1, "Ta": 30, "Tg": 30, "Va": 1.00, "RH": 45, "M": 450,
        "Icl": 0.5, "Ap": 0.0, "Fr": 0.0
    },
    # Case 5
    {
        "accl": 1, "posture": 2, "Ta": 35, "Tg": 50, "Va": 1.00, "RH": 30, "M": 250,
        "Icl": 1.0, "Ap": 0.20, "Fr": 0.85
    }
]

# Common parameters
defspeed = 0
Walksp = 0
defdir = 0
THETA = 0
D = 15  # distance for Tr calc

print("===============================================")

for i, case in enumerate(cases, start=1):
    Ta = case["Ta"]
    Tg = case["Tg"]
    Va = case["Va"]
    RH = case["RH"]

    # Compute Tr (mean radiant temperature)
    Tr = ((Tg + 273) ** 4 + (1.111e8 / (0.95 * (0.01 * D) ** 0.4)) * (Va ** 0.6) * (Tg - Ta)) ** 0.25 - 273
    # Compute vapour pressure Pa
    Pa = 0.01 * RH * 0.6105 * pow(2.71828, 17.27 * Ta / (237.3 + Ta))

    Tre, SWp, SWtotg, Dlimtcr, Dlimloss = PHS79332023(
        Ta, Tr, Pa, Va, case["M"], case["posture"], case["Icl"],
        case["Ap"], case["Fr"], defspeed, Walksp, defdir, THETA, case["accl"]
    )

    print(f"Case {i}")
    print(f"t_re: {Tre:.1f}")
    print(f"water_loss (g): {SWtotg:.1f}")
    print(f"d_lim_loss_95: {Dlimloss:.1f}")
    print(f"d_lim_t_re: {Dlimtcr:.1f}")
    print("===============================================")
