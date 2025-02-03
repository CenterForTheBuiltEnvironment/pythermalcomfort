from dataclasses import dataclass, field
from enum import Enum
from typing import Union

import numpy as np

from pythermalcomfort.utilities import Postures, Sex, Units, validate_type


@dataclass
class BaseInputs:
    """Base class containing all possible input parameters."""

    body_surface_area: Union[float, int, np.ndarray, list] = field(default=1.8258)
    tdb: Union[float, int, np.ndarray, list] = field(default=None)
    tr: Union[float, int, np.ndarray, list] = field(default=None)
    twb: Union[float, int, np.ndarray, list] = field(default=None)
    tg: Union[float, int, np.ndarray, list] = field(default=None)
    vr: Union[float, int, np.ndarray, list] = field(default=None)
    v: Union[float, int, np.ndarray, list] = field(default=None)
    rh: Union[float, int, np.ndarray, list] = field(default=None)
    met: Union[float, int, np.ndarray, list] = field(default=None)
    clo: Union[float, int, np.ndarray, list] = field(default=None)
    wme: Union[float, int, np.ndarray, list] = field(default=0)
    round_output: bool = field(default=True)
    limit_inputs: bool = field(default=True)
    with_solar_load: bool = field(default=False)
    airspeed_control: bool = field(default=True)
    units: str = field(default=Units.SI.value)
    a_coefficient: Union[float, int] = field(default=None)
    e_coefficient: Union[float, int] = field(default=None)
    v_ankle: Union[float, int, np.ndarray, list] = field(default=None)
    t_running_mean: Union[float, int, np.ndarray, list] = field(default=None)
    q: Union[float, int, np.ndarray, list] = field(default=None)
    tout: Union[float, int, np.ndarray, list] = field(default=None)
    p_atm: Union[float, int, np.ndarray, list] = field(default=101325)
    age: Union[float, int, np.ndarray, list] = field(default=None)
    weight: Union[float, int, np.ndarray, list] = field(default=None)
    height: Union[float, int, np.ndarray, list] = field(default=None)
    sol_altitude: Union[float, int, np.ndarray, list] = field(default=None)
    sharp: Union[float, int, np.ndarray, list] = field(default=None)
    sol_radiation_dir: Union[float, int, np.ndarray, list] = field(default=None)
    sol_transmittance: Union[float, int, np.ndarray, list] = field(default=None)
    f_svv: Union[float, int, np.ndarray, list] = field(default=None)
    f_bes: Union[float, int, np.ndarray, list] = field(default=None)
    asw: Union[float, int, np.ndarray, list] = field(default=None)
    floor_reflectance: Union[float, int, np.ndarray, list] = field(default=None)
    vertical_tmp_grad: Union[float, int, np.ndarray, list] = field(default=None)
    position: Union[str, np.ndarray, list] = field(default=None)
    sex: Union[str, np.ndarray, list] = field(default=None)
    posture: Union[str, np.ndarray, list] = field(default=None)
    max_skin_blood_flow: Union[float, int, np.ndarray, list] = field(default=80)
    max_sweating: Union[float, int, np.ndarray, list] = field(default=500)
    w_max: Union[float, int, np.ndarray, list] = field(default=None)

    def __post_init__(self):
        def is_pandas_series(obj):
            return type(obj).__name__ == "Series"

        def convert_series_to_list(obj):
            return obj.tolist() if is_pandas_series(obj) else obj

        # Only validate attributes that are not None
        if self.units.upper() not in [Units.SI.value, Units.IP.value]:
            raise ValueError("Units must be either 'SI' or 'IP'")
        if self.position is not None:
            if self.position.lower() not in [
                Postures.sitting.value,
                Postures.standing.value,
                "standing, forced convection",
            ]:
                raise ValueError(
                    "position must be either 'standing', 'sitting', or 'standing, "
                    "forced convection'"
                )
        if self.posture is not None:
            if self.posture.lower() not in [
                Postures.sitting.value,
                Postures.standing.value,
                Postures.crouching.value,
            ]:
                raise ValueError(
                    "posture must be either 'sitting', 'standing', or 'crouching'"
                )
        if self.sex is not None:
            if self.sex.lower() not in [Sex.male.value, Sex.female.value]:
                raise ValueError("sex must be either 'male' or 'female'")
        if self.tdb is not None:
            self.tdb = convert_series_to_list(self.tdb)
            validate_type(self.tdb, "tdb", (float, int, np.ndarray, list))
        if self.tr is not None:
            self.tr = convert_series_to_list(self.tr)
            validate_type(self.tr, "tr", (float, int, np.ndarray, list))
        if self.twb is not None:
            self.twb = convert_series_to_list(self.twb)
            validate_type(self.twb, "twb", (float, int, np.ndarray, list))
        if self.tg is not None:
            self.tg = convert_series_to_list(self.tg)
            validate_type(self.tg, "tg", (float, int, np.ndarray, list))
        if self.vr is not None:
            self.vr = convert_series_to_list(self.vr)
            validate_type(self.vr, "vr", (float, int, np.ndarray, list))
        if self.v is not None:
            self.v = convert_series_to_list(self.v)
            validate_type(self.v, "v", (float, int, np.ndarray, list))
        if self.rh is not None:
            self.rh = convert_series_to_list(self.rh)
            validate_type(self.rh, "rh", (float, int, np.ndarray, list))
        if self.met is not None:
            self.met = convert_series_to_list(self.met)
            validate_type(self.met, "met", (float, int, np.ndarray, list))
        if self.clo is not None:
            self.clo = convert_series_to_list(self.clo)
            validate_type(self.clo, "clo", (float, int, np.ndarray, list))
        if self.a_coefficient is not None:
            self.a_coefficient = convert_series_to_list(self.a_coefficient)
            validate_type(self.a_coefficient, "a_coefficient", (float, int))
        if self.e_coefficient is not None:
            self.e_coefficient = convert_series_to_list(self.e_coefficient)
            validate_type(self.e_coefficient, "e_coefficient", (float, int))
        if self.wme is not None:
            self.wme = convert_series_to_list(self.wme)
            validate_type(self.wme, "wme", (float, int, np.ndarray, list))
        if self.v_ankle is not None:
            self.v_ankle = convert_series_to_list(self.v_ankle)
            validate_type(self.v_ankle, "v_ankle", (float, int, np.ndarray, list))
        if self.t_running_mean is not None:
            self.t_running_mean = convert_series_to_list(self.t_running_mean)
            validate_type(
                self.t_running_mean, "t_running_mean", (float, int, np.ndarray, list)
            )
        if self.q is not None:
            self.q = convert_series_to_list(self.q)
            validate_type(self.q, "q", (float, int, np.ndarray, list))
        if not isinstance(self.round_output, bool):
            raise TypeError("round must be either True or False.")
        if not isinstance(self.limit_inputs, bool):
            raise TypeError("limit_inputs must be either True or False.")
        if not isinstance(self.airspeed_control, bool):
            raise TypeError("airspeed_control must be either True or False.")
        if not isinstance(self.with_solar_load, bool):
            raise TypeError("with_solar_load must be either True or False.")
        if self.tout is not None:
            self.tout = convert_series_to_list(self.tout)
            validate_type(self.tout, "tout", (float, int, np.ndarray, list))
        if self.p_atm is not None:
            self.p_atm = convert_series_to_list(self.p_atm)
            validate_type(self.p_atm, "p_atm", (float, int, np.ndarray, list))
        if self.age is not None:
            self.age = convert_series_to_list(self.age)
            validate_type(self.age, "age", (float, int, np.ndarray, list))
        if self.weight is not None:
            self.weight = convert_series_to_list(self.weight)
            validate_type(self.weight, "weight", (float, int, np.ndarray, list))
        if self.height is not None:
            self.height = convert_series_to_list(self.height)
            validate_type(self.height, "height", (float, int, np.ndarray, list))
        if self.sol_altitude is not None:
            self.sol_altitude = convert_series_to_list(self.sol_altitude)
            validate_type(
                self.sol_altitude, "sol_altitude", (float, int, np.ndarray, list)
            )
        if self.sharp is not None:
            self.sharp = convert_series_to_list(self.sharp)
            validate_type(self.sharp, "sharp", (float, int, np.ndarray, list))
        if self.sol_radiation_dir is not None:
            self.sol_radiation_dir = convert_series_to_list(self.sol_radiation_dir)
            validate_type(
                self.sol_radiation_dir,
                "sol_radiation_dir",
                (float, int, np.ndarray, list),
            )
        if self.sol_transmittance is not None:
            self.sol_transmittance = convert_series_to_list(self.sol_transmittance)
            validate_type(
                self.sol_transmittance,
                "sol_transmittance",
                (float, int, np.ndarray, list),
            )
        if self.f_svv is not None:
            self.f_svv = convert_series_to_list(self.f_svv)
            validate_type(self.f_svv, "f_svv", (float, int, np.ndarray, list))
        if self.f_bes is not None:
            self.f_bes = convert_series_to_list(self.f_bes)
            validate_type(self.f_bes, "f_bes", (float, int, np.ndarray, list))
        if self.asw is not None:
            self.asw = convert_series_to_list(self.asw)
            validate_type(self.asw, "asw", (float, int, np.ndarray, list))
        if self.floor_reflectance is not None:
            self.floor_reflectance = convert_series_to_list(self.floor_reflectance)
            validate_type(
                self.floor_reflectance,
                "floor_reflectance",
                (float, int, np.ndarray, list),
            )
        if self.vertical_tmp_grad is not None:
            self.vertical_tmp_grad = convert_series_to_list(self.vertical_tmp_grad)
            validate_type(
                self.vertical_tmp_grad,
                "vertical_tmp_grad",
                (float, int, np.ndarray, list),
            )
        if self.body_surface_area is not None:
            self.body_surface_area = convert_series_to_list(self.body_surface_area)
            validate_type(
                self.body_surface_area,
                "body_surface_area",
                (float, int, np.ndarray, list),
            )
        if self.max_sweating is not None:
            self.max_sweating = convert_series_to_list(self.max_sweating)
            validate_type(self.max_sweating, "max_sweating", (float, int))
        if self.max_skin_blood_flow is not None:
            self.max_skin_blood_flow = convert_series_to_list(self.max_skin_blood_flow)
            validate_type(self.max_skin_blood_flow, "max_skin_blood_flow", (float, int))
        if self.w_max is not None:
            self.w_max = convert_series_to_list(self.w_max)
            validate_type(self.w_max, "w_max", (float, int, np.ndarray, list))


@dataclass
class APMVInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        vr,
        rh,
        met,
        clo,
        a_coefficient,
        wme=0,
        units=Units.SI.value,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            tr=tr,
            vr=vr,
            rh=rh,
            met=met,
            clo=clo,
            a_coefficient=a_coefficient,
            wme=wme,
            units=units,
        )


@dataclass
class ASHRAEInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        t_running_mean,
        v,
        units,
    ):
        super().__init__(
            tdb=tdb,
            tr=tr,
            v=v,
            units=units,
            t_running_mean=t_running_mean,
        )


@dataclass
class ENInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        t_running_mean,
        v,
        units,
    ):
        super().__init__(
            tdb=tdb,
            tr=tr,
            v=v,
            units=units,
            t_running_mean=t_running_mean,
        )


@dataclass
class AnkleDraftInputs(BaseInputs):
    """Child class that only requires specific attributes."""

    def __init__(
        self,
        tdb,
        tr,
        vr,
        rh,
        met,
        clo,
        v_ankle,
        units=Units.SI.value,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            tr=tr,
            vr=vr,
            rh=rh,
            met=met,
            clo=clo,
            v_ankle=v_ankle,
            units=units,
        )


@dataclass
class ATInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        rh,
        v,
        q=None,
        round_output=True,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            vr=v,
            rh=rh,
            q=q,
            round_output=round_output,
        )


@dataclass
class ATHBInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        vr,
        rh,
        met,
        t_running_mean,
    ):
        super().__init__(
            tdb=tdb,
            tr=tr,
            vr=vr,
            rh=rh,
            met=met,
            t_running_mean=t_running_mean,
        )


@dataclass
class CloTOutInputs(BaseInputs):
    def __init__(
        self,
        tout,
        units: str = Units.SI.value,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tout=tout,
            units=units,
        )


@dataclass
class CEInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        vr,
        rh,
        met,
        clo,
        wme,
        units,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            tr=tr,
            vr=vr,
            rh=rh,
            met=met,
            clo=clo,
            wme=wme,
            units=units,
        )


@dataclass
class DIInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        rh,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            rh=rh,
        )


@dataclass
class EPMVInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        vr,
        rh,
        met,
        clo,
        e_coefficient,
        wme,
        units,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            tr=tr,
            vr=vr,
            rh=rh,
            met=met,
            clo=clo,
            e_coefficient=e_coefficient,
            wme=wme,
            units=units,
        )


class HIModels(Enum):
    rothfusz = "rothfusz"
    lu_romps = "lu-romps"


@dataclass
class HIInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        rh,
        round_output,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            rh=rh,
            round_output=round_output,
        )


class HumidexModels(Enum):
    rana = "rana"
    masterson = "masterson"


@dataclass
class HumidexInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        rh,
        round_output,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            rh=rh,
            round_output=round_output,
        )


@dataclass
class NETInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        rh,
        v,
        round_output=True,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            rh=rh,
            v=v,
            round_output=round_output,
        )


@dataclass
class PETSteadyInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        v,
        rh,
        met,
        clo,
        p_atm,
        position,
        age,
        sex,
        weight,
        height,
        wme,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            tr=tr,
            v=v,
            rh=rh,
            met=met,
            clo=clo,
            p_atm=p_atm,
            position=position,
            age=age,
            sex=sex,
            weight=weight,
            height=height,
            wme=wme,
        )


@dataclass
class PHSInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        v,
        rh,
        met,
        clo,
        round_output,
        wme,
        posture,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            tr=tr,
            v=v,
            rh=rh,
            met=met,
            clo=clo,
            round_output=round_output,
            wme=wme,
            posture=posture,
        )


@dataclass
class PMVInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        vr,
        rh,
        met,
        clo,
        wme=0,
        units=Units.SI.value,
        limit_inputs=True,
        airspeed_control=True,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            tr=tr,
            vr=vr,
            rh=rh,
            met=met,
            clo=clo,
            wme=wme,
            units=units,
            limit_inputs=limit_inputs,
            airspeed_control=airspeed_control,
        )


@dataclass
class PMVPPDInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        vr,
        rh,
        met,
        clo,
        wme=0,
        units=Units.SI.value,
        limit_inputs=True,
        airspeed_control=True,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            tr=tr,
            vr=vr,
            rh=rh,
            met=met,
            clo=clo,
            wme=wme,
            units=units,
            limit_inputs=limit_inputs,
            airspeed_control=airspeed_control,
        )


@dataclass
class SETInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        v,
        rh,
        met,
        clo,
        wme=0,
        body_surface_area=1.8258,
        p_atm=101325,
        position=Postures.standing.value,
        units=Units.SI.value,
        limit_inputs=True,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            tr=tr,
            v=v,
            rh=rh,
            met=met,
            clo=clo,
            wme=wme,
            body_surface_area=body_surface_area,
            p_atm=p_atm,
            position=position,
            units=units,
            limit_inputs=limit_inputs,
        )


@dataclass
class SolarGainInputs(BaseInputs):
    def __init__(
        self,
        sol_altitude,
        sharp,
        sol_radiation_dir,
        sol_transmittance,
        f_svv,
        f_bes,
        asw=0.7,
        posture=Postures.sitting.value,
        floor_reflectance=0.6,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            sol_altitude=sol_altitude,
            sharp=sharp,
            sol_radiation_dir=sol_radiation_dir,
            sol_transmittance=sol_transmittance,
            f_svv=f_svv,
            f_bes=f_bes,
            asw=asw,
            posture=posture,
            floor_reflectance=floor_reflectance,
        )


@dataclass
class GaggeTwoNodesInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        v,
        rh,
        met,
        clo,
        wme=0,
        body_surface_area=1.8258,
        p_atm=101325,
        position=Postures.standing.value,
        max_skin_blood_flow=90,
        round_output=True,
        max_sweating=500,
        w_max=None,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            tr=tr,
            v=v,
            rh=rh,
            met=met,
            clo=clo,
            wme=wme,
            body_surface_area=body_surface_area,
            p_atm=p_atm,
            position=position,
            max_skin_blood_flow=max_skin_blood_flow,
            round_output=round_output,
            max_sweating=max_sweating,
            w_max=w_max,
        )


@dataclass
class UseFansHeatwavesInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        v,
        rh,
        met,
        clo,
        wme=0,
        body_surface_area=1.8258,
        p_atm=101325,
        position=Postures.standing.value,
        max_skin_blood_flow=80,
        limit_inputs=True,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            tr=tr,
            v=v,
            rh=rh,
            met=met,
            clo=clo,
            wme=wme,
            body_surface_area=body_surface_area,
            p_atm=p_atm,
            position=position,
            max_skin_blood_flow=max_skin_blood_flow,
            limit_inputs=limit_inputs,
        )


@dataclass
class UTCIInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        v,
        rh,
        units=Units.SI.value,
        limit_inputs=True,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            tr=tr,
            v=v,
            rh=rh,
            units=units,
            limit_inputs=limit_inputs,
        )


@dataclass
class VerticalTGradPPDInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        tr,
        vr,
        rh,
        met,
        clo,
        vertical_tmp_grad,
        units=Units.SI.value,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            tr=tr,
            vr=vr,
            rh=rh,
            met=met,
            clo=clo,
            vertical_tmp_grad=vertical_tmp_grad,
            units=units,
        )


@dataclass
class WBGTInputs(BaseInputs):
    def __init__(
        self,
        twb,
        tg,
        tdb=None,
        with_solar_load=False,
        round_output=True,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            twb=twb,
            tg=tg,
            tdb=tdb,
            with_solar_load=with_solar_load,
            round_output=round_output,
        )


@dataclass
class WCIInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        v,
        round_output=True,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            v=v,
            round_output=round_output,
        )


@dataclass
class WCTInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        v,
        round_output=True,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            v=v,
            round_output=round_output,
        )
