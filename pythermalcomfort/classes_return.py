import datetime as dt
from dataclasses import dataclass, fields
from typing import Optional, Union

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class APMV:
    """A dataclass to store the results of the adaptive Predicted Mean Vote (aPMV)
    model.

    Attributes
    ----------
    a_pmv : float or list of floats
        Predicted Mean Vote.
    """

    a_pmv: Union[float, npt.ArrayLike]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass(frozen=True)
class AdaptiveASHRAE:
    """A dataclass to store the results of the adaptive thermal comfort model based on
    ASHRAE 55.

    Attributes
    ----------
    tmp_cmf : float or list of floats
        Comfort temperature at a specific running mean temperature, default in [°C] or [°F].
    tmp_cmf_80_low : float or list of floats
        Lower acceptable comfort temperature for 80% occupants, default in [°C] or [°F].
    tmp_cmf_80_up : float or list of floats
        Upper acceptable comfort temperature for 80% occupants, default in [°C] or [°F].
    tmp_cmf_90_low : float or list of floats
        Lower acceptable comfort temperature for 90% occupants, default in [°C] or [°F].
    tmp_cmf_90_up : float or list of floats
        Upper acceptable comfort temperature for 90% occupants, default in [°C] or [°F].
    acceptability_80 : bool or list of bools
        Acceptability for 80% occupants.
    acceptability_90 : bool or list of bools
        Acceptability for 90% occupants.
    """

    tmp_cmf: Union[float, npt.ArrayLike]
    tmp_cmf_80_low: Union[float, npt.ArrayLike]
    tmp_cmf_80_up: Union[float, npt.ArrayLike]
    tmp_cmf_90_low: Union[float, npt.ArrayLike]
    tmp_cmf_90_up: Union[float, npt.ArrayLike]
    acceptability_80: Union[bool, npt.ArrayLike]
    acceptability_90: Union[bool, npt.ArrayLike]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass
class AdaptiveEN:
    """Dataclass to store the results of the adaptive thermal comfort calculation based
    on EN 16798-1 2019.

    Attributes
    ----------
    tmp_cmf : float or list of floats
        Comfort temperature at that specific running mean temperature, default in [°C] or in [°F].
    acceptability_cat_i : bool or list of bools
        If the indoor conditions comply with comfort category I.
    acceptability_cat_ii : bool or list of bools
        If the indoor conditions comply with comfort category II.
    acceptability_cat_iii : bool or list of bools
        If the indoor conditions comply with comfort category III.
    tmp_cmf_cat_i_up : float or list of floats
        Upper acceptable comfort temperature for category I, default in [°C] or in [°F].
    tmp_cmf_cat_ii_up : float or list of floats
        Upper acceptable comfort temperature for category II, default in [°C] or in [°F].
    tmp_cmf_cat_iii_up : float or list of floats
        Upper acceptable comfort temperature for category III, default in [°C] or in [°F].
    tmp_cmf_cat_i_low : float or list of floats
        Lower acceptable comfort temperature for category I, default in [°C] or in [°F].
    tmp_cmf_cat_ii_low : float or list of floats
        Lower acceptable comfort temperature for category II, default in [°C] or in [°F].
    tmp_cmf_cat_iii_low : float or list of floats
        Lower acceptable comfort temperature for category III, default in [°C] or in [°F].
    """

    tmp_cmf: Union[float, npt.ArrayLike]
    acceptability_cat_i: Union[bool, npt.ArrayLike]
    acceptability_cat_ii: Union[bool, npt.ArrayLike]
    acceptability_cat_iii: Union[bool, npt.ArrayLike]
    tmp_cmf_cat_i_up: Union[float, npt.ArrayLike]
    tmp_cmf_cat_ii_up: Union[float, npt.ArrayLike]
    tmp_cmf_cat_iii_up: Union[float, npt.ArrayLike]
    tmp_cmf_cat_i_low: Union[float, npt.ArrayLike]
    tmp_cmf_cat_ii_low: Union[float, npt.ArrayLike]
    tmp_cmf_cat_iii_low: Union[float, npt.ArrayLike]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass(frozen=True)
class AnkleDraft:
    """Dataclass to store the results of the ankle draft calculation.

    Attributes
    ----------
    ppd_ad : float or list of floats
        Predicted Percentage of Dissatisfied occupants with ankle draft, [%].
    acceptability : bool or list of bools
        Indicates if the air speed at the ankle level is acceptable according to ASHRAE 55 2020 standard.
    """

    ppd_ad: Union[float, npt.ArrayLike]
    acceptability: Union[bool, npt.ArrayLike]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass(frozen=True)
class AT:
    """Dataclass to store the results of the Apparent Temperature (AT) calculation.

    Attributes
    ----------
    at : float or list of floats
        Apparent temperature, [°C]
    """

    at: float

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass(frozen=True)
class ATHB:
    """Dataclass to store the results of the Adaptive Thermal Heat Balance (ATHB)
    calculation.

    Attributes
    ----------
    athb_pmv : float or list of floats
        Predicted Mean Vote calculated with the Adaptive Thermal Heat Balance framework.
    """

    athb_pmv: Union[float, npt.ArrayLike]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass(frozen=True)
class CloTOut:
    """Dataclass to represent the clothing insulation Icl as a function of outdoor air
    temperature.

    Attributes
    ----------
    clo_tout : float or list of floats
        Representative clothing insulation Icl.
    """

    clo_tout: Union[float, list[float]]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass(frozen=True)
class CE:
    """Dataclass to represent the Cooling Effect (CE).

    Attributes
    ----------
    ce : float or list of floats
        Cooling Effect value.
    """

    ce: Union[float, list[float]]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass(frozen=True)
class DI:
    """Dataclass to represent the Discomfort Index (DI) and its classification.

    Attributes
    ----------
    di : float or list of floats
        Discomfort Index, [°C].
    discomfort_condition : str or list of str
        Classification of the thermal comfort conditions according to the discomfort index.
    """

    di: Union[float, list[float]]
    discomfort_condition: Union[str, list[str]]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass(frozen=True)
class EPMV:
    """Dataclass to represent the Adjusted Predicted Mean Votes with Expectancy Factor
    (ePMV).

    Attributes
    ----------
    e_pmv : float or list of floats
        Adjusted Predicted Mean Votes with Expectancy Factor.
    """

    e_pmv: Union[float, list[float]]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass(frozen=True)
class HI:
    """Dataclass to represent the Heat Index (HI).

    Attributes
    ----------
    hi : float or list of floats
        Heat Index, [°C] or [°F] depending on the units.
    """

    hi: Union[float, list[float]]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass(frozen=True)
class Humidex:
    """Dataclass to represent the Humidex and its discomfort category.

    Attributes
    ----------
    humidex : float or list of floats
        Humidex value, [°C].
    discomfort : str or list of str
        Degree of comfort or discomfort as defined in Havenith and Fiala (2016).
    """

    humidex: Union[float, list[float]]
    discomfort: Union[str, list[str]]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass(frozen=True)
class NET:
    """Dataclass to represent the Normal Effective Temperature (NET).

    Attributes
    ----------
    net : float or list of floats
        Normal Effective Temperature, [°C].
    """

    net: Union[float, list[float]]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass(frozen=True)
class PETSteady:
    """Dataclass to represent the Physiological Equivalent Temperature (PET).

    Attributes
    ----------
    pet : float or list of floats
        Physiological Equivalent Temperature.
    """

    pet: Union[float, list[float]]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass(frozen=True)
class PHS:
    """Dataclass to represent the Predicted Heat Strain (PHS).

    Attributes
    ----------
    t_re : float or list of floats
        Rectal temperature, [°C].
    t_sk : float or list of floats
        Skin temperature, [°C].
    t_cr : float or list of floats
        Core temperature, [°C].
    t_cr_eq : float or list of floats
        Core temperature as a function of the metabolic rate, [°C].
    t_sk_t_cr_wg : float or list of floats
        Fraction of the body mass at the skin temperature.
    d_lim_loss_50 : float or list of floats
        Maximum allowable exposure time for water loss, mean subject, [minutes].
    d_lim_loss_95 : float or list of floats
        Maximum allowable exposure time for water loss, 95% of the working population, [minutes].
    d_lim_t_re : float or list of floats
        Maximum allowable exposure time for heat storage, [minutes].
    water_loss_watt : float or list of floats
        Maximum water loss in watts, [W].
    water_loss : float or list of floats
        Maximum water loss, [g].
    """

    t_re: Union[float, list[float]]
    t_sk: Union[float, list[float]]
    t_cr: Union[float, list[float]]
    t_cr_eq: Union[float, list[float]]
    t_sk_t_cr_wg: Union[float, list[float]]
    d_lim_loss_50: Union[float, list[float]]
    d_lim_loss_95: Union[float, list[float]]
    d_lim_t_re: Union[float, list[float]]
    water_loss_watt: Union[float, list[float]]
    water_loss: Union[float, list[float]]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass(frozen=True)
class PMV:
    """Dataclass to represent the Predicted Mean Vote (PMV).

    Attributes
    ----------
    pmv : float or list of floats
        Predicted Mean Vote.
    """

    pmv: Union[float, npt.ArrayLike]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass(frozen=True)
class PMVPPD:
    """Dataclass to represent the Predicted Mean Vote (PMV) and Predicted Percentage of
    Dissatisfied (PPD).

    Attributes
    ----------
    pmv : float or list of floats
        Predicted Mean Vote.
    ppd : float or list of floats
        Predicted Percentage of Dissatisfied.
    tsv : str or list of strings
        Predicted thermal sensation vote.
    """

    pmv: Union[float, list[float]]
    ppd: Union[float, list[float]]
    tsv: Union[float, list[float]]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass(frozen=True)
class PsychrometricValues:
    p_sat: Union[float, list[float]]
    p_vap: Union[float, list[float]]
    hr: Union[float, list[float]]
    wet_bulb_tmp: Union[float, list[float]]
    dew_point_tmp: Union[float, list[float]]
    h: Union[float, list[float]]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass(frozen=True)
class SET:
    """Dataclass to represent the Standard Effective Temperature (SET).

    Attributes
    ----------
    set : float or list of floats
        Standard effective temperature, [°C].
    """

    set: Union[float, list[float]]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass(frozen=True)
class SolarGain:
    """Dataclass to represent the solar gain to the human body.

    Attributes
    ----------
    erf : float or list of floats
        Solar gain to the human body using the Effective Radiant Field [W/m2].
    delta_mrt : float or list of floats
        Delta mean radiant temperature. The amount by which the mean radiant
        temperature of the space should be increased if no solar radiation is present.
    """

    erf: Union[float, list[float]]
    delta_mrt: Union[float, list[float]]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass(frozen=True)
class GaggeTwoNodes:
    """Dataclass to represent the results of the two-node model of human temperature
    regulation.

    Attributes
    ----------
    e_skin : float or list of floats
        Total rate of evaporative heat loss from skin, [W/m2]. Equal to e_rsw + e_diff.
    e_rsw : float or list of floats
        Rate of evaporative heat loss from sweat evaporation, [W/m2].
    e_max : float or list of floats
        Maximum rate of evaporative heat loss from skin, [W/m2].
    q_sensible : float or list of floats
        Sensible heat loss from skin, [W/m2].
    q_skin : float or list of floats
        Total rate of heat loss from skin, [W/m2]. Equal to q_sensible + e_skin.
    q_res : float or list of floats
        Total rate of heat loss through respiration, [W/m2].
    t_core : float or list of floats
        Core temperature, [°C].
    t_skin : float or list of floats
        Skin temperature, [°C].
    m_bl : float or list of floats
        Skin blood flow, [kg/h/m2].
    m_rsw : float or list of floats
        Rate at which regulatory sweat is generated, [mL/h/m2].
    w : float or list of floats
        Skin wettedness, adimensional. Ranges from 0 to 1.
    w_max : float or list of floats
        Skin wettedness (w) practical upper limit, adimensional. Ranges from 0 to 1.
    set : float or list of floats
        Standard Effective Temperature (SET).
    et : float or list of floats
        New Effective Temperature (ET).
    pmv_gagge : float or list of floats
        PMV Gagge.
    pmv_set : float or list of floats
        PMV SET.
    disc : float or list of floats
        Thermal discomfort.
    t_sens : float or list of floats
        Predicted Thermal Sensation.
    """

    e_skin: Union[float, list[float]]
    e_rsw: Union[float, list[float]]
    e_max: Union[float, list[float]]
    q_sensible: Union[float, list[float]]
    q_skin: Union[float, list[float]]
    q_res: Union[float, list[float]]
    t_core: Union[float, list[float]]
    t_skin: Union[float, list[float]]
    m_bl: Union[float, list[float]]
    m_rsw: Union[float, list[float]]
    w: Union[float, list[float]]
    w_max: Union[float, list[float]]
    set: Union[float, list[float]]
    et: Union[float, list[float]]
    pmv_gagge: Union[float, list[float]]
    pmv_set: Union[float, list[float]]
    disc: Union[float, list[float]]
    t_sens: Union[float, list[float]]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass(frozen=True)
class UseFansHeatwaves:
    """Dataclass to represent the results of using fans during heatwaves.

    Attributes
    ----------
    e_skin : float or list of floats
        Total rate of evaporative heat loss from skin, [W/m2]. Equal to e_rsw + e_diff.
    e_rsw : float or list of floats
        Rate of evaporative heat loss from sweat evaporation, [W/m2].
    e_max : float or list of floats
        Maximum rate of evaporative heat loss from skin, [W/m2].
    q_sensible : float or list of floats
        Sensible heat loss from skin, [W/m2].
    q_skin : float or list of floats
        Total rate of heat loss from skin, [W/m2]. Equal to q_sensible + e_skin.
    q_res : float or list of floats
        Total rate of heat loss through respiration, [W/m2].
    t_core : float or list of floats
        Core temperature, [°C].
    t_skin : float or list of floats
        Skin temperature, [°C].
    m_bl : float or list of floats
        Skin blood flow, [kg/h/m2].
    m_rsw : float or list of floats
        Rate at which regulatory sweat is generated, [mL/h/m2].
    w : float or list of floats
        Skin wettedness, adimensional. Ranges from 0 to 1.
    w_max : float or list of floats
        Skin wettedness (w) practical upper limit, adimensional. Ranges from 0 to 1.
    heat_strain : bool or list of bools
        True if the model predicts that the person may be experiencing heat strain.
    heat_strain_blood_flow : bool or list of bools
        True if heat strain is caused by skin blood flow (m_bl) reaching its maximum value.
    heat_strain_w : bool or list of bools
        True if heat strain is caused by skin wettedness (w) reaching its maximum value.
    heat_strain_sweating : bool or list of bools
        True if heat strain is caused by regulatory sweating (m_rsw) reaching its maximum value.
    """

    e_skin: Union[float, list[float]]
    e_rsw: Union[float, list[float]]
    e_max: Union[float, list[float]]
    q_sensible: Union[float, list[float]]
    q_skin: Union[float, list[float]]
    q_res: Union[float, list[float]]
    t_core: Union[float, list[float]]
    t_skin: Union[float, list[float]]
    m_bl: Union[float, list[float]]
    m_rsw: Union[float, list[float]]
    w: Union[float, list[float]]
    w_max: Union[float, list[float]]
    heat_strain: Union[bool, list[bool]]
    heat_strain_blood_flow: Union[bool, list[bool]]
    heat_strain_w: Union[bool, list[bool]]
    heat_strain_sweating: Union[bool, list[bool]]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass(frozen=True)
class UTCI:
    """Dataclass to represent the Universal Thermal Climate Index (UTCI).

    Attributes
    ----------
    utci : float or list of floats
        Universal Thermal Climate Index, [°C] or in [°F].
    stress_category : str or list of strs
        UTCI categorized in terms of thermal stress [Blazejczyk2013]_.
    """

    utci: Union[float, list[float]]
    stress_category: Union[str, list[str]]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass(frozen=True)
class VerticalTGradPPD:
    """Dataclass to represent the Predicted Percentage of Dissatisfied (PPD) with
    vertical temperature gradient.

    Attributes
    ----------
    ppd_vg : float or list of floats
        Predicted Percentage of Dissatisfied occupants with vertical temperature gradient.
    acceptability : bool or list of bools
        True if the value of air speed at the ankle level is acceptable (PPD_vg <= 5%).
    """

    ppd_vg: Union[float, list[float]]
    acceptability: Union[bool, list[bool]]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass(frozen=True)
class WBGT:
    """Dataclass to represent the Wet Bulb Globe Temperature (WBGT) index.

    Attributes
    ----------
    wbgt : float or list of floats
        Wet Bulb Globe Temperature Index.
    """

    wbgt: Union[float, npt.ArrayLike]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass(frozen=True)
class WCI:
    """Dataclass to represent the Wind Chill Index (WCI).

    Attributes
    ----------
    wci : float or list of floats
        Wind Chill Index, [W/m^2].
    """

    wci: Union[float, list[float]]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass(frozen=True)
class WCT:
    """Dataclass to represent the Wind Chill Temperature (WCT).

    Attributes
    ----------
    wct : float or list of floats
        Wind Chill Temperature, [°C].
    """

    wct: Union[float, list[float]]

    def __getitem__(self, item):
        return getattr(self, item)


@dataclass(frozen=True)
class JOS3BodyParts:
    """Dataclass to represent the body parts in the JOS3 model. It is very important to
    keep the order of the attributes as they are defined in the dataclass ['head',
    'neck', 'chest', 'back', 'pelvis', 'left_shoulder', 'left_arm', 'left_hand',
    'right_shoulder', 'right_arm', 'right_hand', 'left_thigh', 'left_leg', 'left_foot',
    'right_thigh', 'right_leg', 'right_foot']

    Attributes
    ----------
    head : float
        Index of the head.
    neck : float
        Index of the neck.
    chest : float
        Index of the chest.
    back : float
        Index of the back.
    pelvis : float
        Index of the pelvis.
    left_shoulder : float
        Index of the left shoulder.
    left_arm : float
        Index of the left arm.
    left_hand : float
        Index of the left hand.
    right_shoulder : float
        Index of the right shoulder.
    right_arm : float
        Index of the right arm.
    right_hand : float
        Index of the right hand.
    left_thigh : float
        Index of the left thigh.
    left_leg : float
        Index of the left leg.
    left_foot : float
        Index of the left foot.
    right_thigh : float
        Index of the right thigh.
    right_leg : float
        Index of the right leg.
    right_foot : float
        Index of the right hand.
    """

    head: Optional[float] = None
    neck: Optional[float] = None
    chest: Optional[float] = None
    back: Optional[float] = None
    pelvis: Optional[float] = None
    left_shoulder: Optional[float] = None
    left_arm: Optional[float] = None
    left_hand: Optional[float] = None
    right_shoulder: Optional[float] = None
    right_arm: Optional[float] = None
    right_hand: Optional[float] = None
    left_thigh: Optional[float] = None
    left_leg: Optional[float] = None
    left_foot: Optional[float] = None
    right_thigh: Optional[float] = None
    right_leg: Optional[float] = None
    right_foot: Optional[float] = None

    def __getitem__(self, item):
        return getattr(self, item)

    @classmethod
    def get_attribute_names(cls):
        return [field.name for field in fields(cls)]


def get_attribute_values(cls):
    return np.array([getattr(cls, field.name) for field in fields(cls)])


@dataclass(frozen=True)
class JOS3Output:
    """Dataclass to represent the output of the JOS3 model simulation.

    Attributes
    ----------
    simulation_time : datetime.timedelta
        The elapsed simulation time.
    dt : int or float
        The time step in seconds.
    t_skin_mean : float
        Mean skin temperature of the whole body [°C].
    t_skin : np.ndarray
        Skin temperatures by the local body segments [°C].
    t_core : np.ndarray
        Core temperatures by the local body segments [°C].
    w_mean : float
        Mean skin wettedness of the whole body [-].
    w : np.ndarray
        Skin wettedness on local body segments [-].
    weight_loss_by_evap_and_res : float
        Weight loss by evaporation and respiration [g/sec].
    cardiac_output : float
        Cardiac output [L/h].
    q_thermogenesis_total : float
        Total thermogenesis [W].
    q_res : float
        Heat loss by respiration [W].
    q_skin2env : np.ndarray
        Total heat loss from the skin to the environment [W].
    height : float
        Body height [m].
    weight : float
        Body weight [kg].
    bsa : np.ndarray
        Body surface area.
    fat : float
        Body fat rate [%].
    sex : str
        Sex.
    age : int
        Age [years].
    t_core_set : np.ndarray
        Core set point temperature [°C].
    t_skin_set : np.ndarray
        Skin set point temperature [°C].
    t_cb : float
        Central blood temperature [°C].
    t_artery : np.ndarray
        Arterial blood temperature [°C].
    t_vein : np.ndarray
        Venous blood temperature [°C].
    t_superficial_vein : np.ndarray
        Superficial venous blood temperature [°C].
    t_muscle : np.ndarray
        Muscle temperature [°C].
    t_fat : np.ndarray
        Fat temperature [°C].
    to : float
        Operative temperature [°C].
    r_t : np.ndarray
        Radiative heat transfer coefficient.
    r_et : np.ndarray
        Evaporative heat transfer coefficient.
    tdb : np.ndarray
        Dry bulb air temperature [°C].
    tr : np.ndarray
        Mean radiant temperature [°C].
    rh : np.ndarray
        Relative humidity [%].
    v : np.ndarray
        Air velocity [m/s].
    par : float
        Physical activity ratio.
    clo : np.ndarray
        Clothing insulation.
    e_skin : np.ndarray
        Evaporative heat loss from the skin [W].
    e_max : np.ndarray
        Maximum evaporative heat loss from the skin [W].
    e_sweat : np.ndarray
        Evaporative heat loss from the skin by only sweating [W].
    bf_core : np.ndarray
        Core blood flow rate [L/h].
    bf_muscle : np.ndarray
        Muscle blood flow rate [L/h].
    bf_fat : np.ndarray
        Fat blood flow rate [L/h].
    bf_skin : np.ndarray
        Skin blood flow rate [L/h].
    bf_ava_hand : np.ndarray
        AVA blood flow rate of one hand [L/h].
    bf_ava_foot : np.ndarray
        AVA blood flow rate of one foot [L/h].
    q_bmr_core : np.ndarray
        Core thermogenesis by basal metabolism [W].
    q_bmr_muscle : np.ndarray
        Muscle thermogenesis by basal metabolism [W].
    q_bmr_fat : np.ndarray
        Fat thermogenesis by basal metabolism [W].
    q_bmr_skin : np.ndarray
        Skin thermogenesis by basal metabolism [W].
    q_work : np.ndarray
        Thermogenesis by work [W].
    q_shiv : np.ndarray
        Thermogenesis by shivering [W].
    q_nst : np.ndarray
        Thermogenesis by non-shivering [W].
    q_thermogenesis_core : np.ndarray
        Core total thermogenesis [W].
    q_thermogenesis_muscle : np.ndarray
        Muscle total thermogenesis [W].
    q_thermogenesis_fat : np.ndarray
        Fat total thermogenesis [W].
    q_thermogenesis_skin : np.ndarray
        Skin total thermogenesis [W].
    q_skin2env_sensible : np.ndarray
        Sensible heat loss from the skin to the environment [W].
    q_skin2env_latent : np.ndarray
        Latent heat loss from the skin to the environment [W].
    q_res_sensible : np.ndarray
        Sensible heat loss by respiration [W].
    q_res_latent : np.ndarray
        Latent heat loss by respiration [W].
    """

    simulation_time: Optional[dt.timedelta] = None
    dt: Optional[float] = None
    t_skin_mean: Optional[float] = None
    t_skin: Optional[JOS3BodyParts] = None
    t_core: Optional[JOS3BodyParts] = None
    w_mean: Optional[float] = None
    w: Optional[JOS3BodyParts] = None
    weight_loss_by_evap_and_res: Optional[float] = None
    cardiac_output: Optional[float] = None
    q_thermogenesis_total: Optional[float] = None
    q_res: Optional[float] = None
    q_skin2env: Optional[JOS3BodyParts] = None
    height: Optional[float] = None
    weight: Optional[float] = None
    bsa: Optional[JOS3BodyParts] = None
    fat: Optional[float] = None
    sex: Optional[str] = None
    age: Optional[int] = None
    t_core_set: Optional[JOS3BodyParts] = None
    t_skin_set: Optional[JOS3BodyParts] = None
    t_cb: Optional[float] = None
    t_artery: Optional[JOS3BodyParts] = None
    t_vein: Optional[JOS3BodyParts] = None
    t_superficial_vein: Optional[JOS3BodyParts] = None
    t_muscle: Optional[JOS3BodyParts] = None
    t_fat: Optional[JOS3BodyParts] = None
    to: Optional[JOS3BodyParts] = None
    r_t: Optional[JOS3BodyParts] = None
    r_et: Optional[JOS3BodyParts] = None
    tdb: Optional[JOS3BodyParts] = None
    tr: Optional[JOS3BodyParts] = None
    rh: Optional[JOS3BodyParts] = None
    v: Optional[JOS3BodyParts] = None
    par: Optional[float] = None
    clo: Optional[JOS3BodyParts] = None
    e_skin: Optional[JOS3BodyParts] = None
    e_max: Optional[JOS3BodyParts] = None
    e_sweat: Optional[JOS3BodyParts] = None
    bf_core: Optional[JOS3BodyParts] = None
    bf_muscle: Optional[JOS3BodyParts] = None
    bf_fat: Optional[JOS3BodyParts] = None
    bf_skin: Optional[JOS3BodyParts] = None
    bf_ava_hand: Optional[float] = None
    bf_ava_foot: Optional[float] = None
    q_bmr_core: Optional[JOS3BodyParts] = None
    q_bmr_muscle: Optional[JOS3BodyParts] = None
    q_bmr_fat: Optional[JOS3BodyParts] = None
    q_bmr_skin: Optional[JOS3BodyParts] = None
    q_work: Optional[JOS3BodyParts] = None
    q_shiv: Optional[JOS3BodyParts] = None
    q_nst: Optional[JOS3BodyParts] = None
    q_thermogenesis_core: Optional[JOS3BodyParts] = None
    q_thermogenesis_muscle: Optional[JOS3BodyParts] = None
    q_thermogenesis_fat: Optional[JOS3BodyParts] = None
    q_thermogenesis_skin: Optional[JOS3BodyParts] = None
    q_skin2env_sensible: Optional[JOS3BodyParts] = None
    q_skin2env_latent: Optional[JOS3BodyParts] = None
    q_res_sensible: Optional[float] = None
    q_res_latent: Optional[float] = None

    def __getitem__(self, item):
        return getattr(self, item)
