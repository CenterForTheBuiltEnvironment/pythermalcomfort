from dataclasses import dataclass
from enum import Enum

from pythermalcomfort.utilities import BaseInputs


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
        units="SI",
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
        units="SI",
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
        units: str = "SI",
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


@dataclass
class HIInputs(BaseInputs):
    def __init__(
        self,
        tdb,
        rh,
        units,
        round_output,
    ):
        # Initialize with only required fields, setting others to None
        super().__init__(
            tdb=tdb,
            rh=rh,
            units=units,
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
        standard="ISO",
        units="SI",
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
            standard=standard,
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
        standard="ISO",
        units="SI",
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
            standard=standard,
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
        position="standing",
        units="SI",
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
        posture="sitting",
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
class TwoNodesInputs(BaseInputs):
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
        position="standing",
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
        position="standing",
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
        units="SI",
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
        units="SI",
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
