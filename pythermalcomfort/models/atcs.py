import logging
import numpy as np
from pythermalcomfort.models.jos3 import JOS3
from pythermalcomfort.jos3_functions.parameters import Default
from pythermalcomfort.models.zhang_comfort import zhang_sensation_comfort
from pythermalcomfort.utilities import DefaultSkinTemperature

# Set logging config as WARNING
logging.basicConfig(level=logging.WARNING)


class ATCS(JOS3):
    """ATCS (Advanced Thermal Comfort Simulator) is an integrated simulator that combines
    JOS-3 model for simulating human thermal physiology, with a thermal sensation and
    comfort model for predicting psychological parameters from the physiological parameters.
    """

    default_iteration_number_to_get_stable_condition = 600

    def __init__(self, *args, **kwargs):
        """Initializes this class by calling the parent class constructor
        and setting initial values for instance variables.

        Parameters
        ----------
        height : float, optional
            body height, in [m]. The default is 1.72.
        weight : float, optional
            body weight, in [kg]. The default is 74.43.
        fat : float, optional
            fat percentage, in [%]. The default is 15.
        age : int, optional
            age, in [years]. The default is 20.
        sex : str, optional
            sex ("male" or "female"). The default is "male".
        ci : float, optional
            Cardiac index, in [L/min/m2]. The default is 2.6432.
        bmr_equation : str, optional
            The equation used to calculate basal metabolic rate (BMR). Choose a BMR equation.
            The default is "harris-benedict" equation created uding Caucasian's data. (DOI: doi.org/10.1073/pnas.4.12.370)
            If the Ganpule's equation (DOI: doi.org/10.1038/sj.ejcn.1602645) for Japanese people is used, input "japanese".
        bsa_equation : str, optional
            The equation used to calculate body surface area (bsa). Choose a bsa equation.
            You can choose "dubois", "fujimoto", "kruazumi", or "takahira". The default is "dubois".
            The body surface area can be calculated using the function
            :py:meth:`pythermalcomfort.utilities.body_surface_area`.
        ex_output : None, list or "all", optional
            This is used when you want to display results other than the default output parameters (ex.skin temperature);
            by default, JOS outputs only the most necessary parameters in order to reduce the computational load.
            If the parameters other than the default output parameters are needed,
            specify the list of the desired parameter names in string format like ["bf_skin", "bf_core", "t_artery"].
            If you want to display all output results, set ex_output is "all".

        Attributes
        ----------
        tdb : float, int, or array-like
            Dry bulb air temperature [°C].
        tr : float, int, or array-like
            Mean radiant temperature [°C].
        to : float, int, or array-like
            Operative temperature [°C].
        v : float, int, or array-like
            Air speed [m/s].
        rh : float, int, or array-like
            Relative humidity [%].
        clo : float, int, or array-like
            Clothing insulation [clo].
            Note: If you want to input clothing insulation to each body part,
            it can be input using the dictionaly in "utilities.py" in "jos3_function" folder.
            :py:meth:`pythermalcomfort.jos3_functions.utilities.local_clo_typical_ensembles`.
        par : float
            Physical activity ratio [-].
            This equals the ratio of metabolic rate to basal metabolic rate.
            The par of sitting quietly is 1.2.
        posture : str
            Body posture [-]. Choose a posture from standing, sitting or lying.
        body_temp : numpy.ndarray (85,)
            All segment temperatures of JOS-3

        Methods
        -------
        simulate(times, dtime, output):
            Run this model for given times.
        dict_results():
            Get results as a dictionary with pandas.DataFrame values.

        Returns
        -------
        Predicted thermal phygiology
            The same output as JOS-3 model
        Predicted thermal sensation and comfort
            Outputs from Zhang's sensation and comfort model with the inputs of JOS-3 model

        Examples
        -------
        .. code-block:: python
        >>> from pythermalcomfort.models import atcs
        >>> import pandas as pd
        >>> import matplotlib.pyplot as plt
        >>> import os
        >>>
        >>># Build a model and set a body built
        >>># Create an instance of the class with optional body parameters such as body height, weight, age, sex, etc.
        >>> model = atcs.ATCS(height=1.7, weight=60, age=30)
        >>>
        >>># Set the first phase
        >>> model.to = 28  # Operative temperature [°C]
        >>> model.rh = 40  # Relative humidity [%]
        >>> model.v = 0.2  # Air velocity [m/s]
        >>> model.par = 1.2  # Physical activity ratio [-]
        >>> model.simulate(60)  # Exposure time = 60 [min]
        >>>
        >>># Set the next condition (You only need to change the parameters that you want to change)
        >>> model.to = 20  # Change only operative temperature
        >>> model.simulate(60)  # Additional exposure time = 60 [min]
        >>>
        >>># Show the results
        >>> df = pd.DataFrame(model.dict_results())
        >>> df["sensation_overall"].plot()
        >>> plt.ylabel("Overall thermal sensation [-]")
        >>> plt.xlabel("Time [min]")
        >>> plt.savefig("atcs_example_local_sensations.png"))
        >>> plt.show()
        >>>
        >>># Exporting the results as csv
        >>> model.to_csv("atcs_example1 (default output).csv"))
        """
        # Initialize instance variables for this class.
        self.cached_clo = np.ones(17) * Default.clothing_insulation
        self.cached_par = Default.physical_activity_ratio
        self.cached_overall_clo = np.average(self.cached_clo, weights=Default.local_bsa)

        # Initialize parent class
        self.comfort_setpt_skin = np.array(list(DefaultSkinTemperature()._asdict()))
        self.t_skin_used = np.ones(17) * Default.skin_temperature
        self.t_skin_wet = np.ones(17) * Default.skin_temperature
        self.t_skin_previous_time_step = None
        self.t_core_previous_time_step = None
        self.overall_clo = Default.clothing_insulation

        # Initialize parent class
        super().__init__(*args, **kwargs)

        self.overall_clo = np.average(self.clo, weights=Default.local_bsa)
        logging.info(f"During initialization, overall clo value is {self.overall_clo}")

    def _calculate_comfort_setpt(self, overall_clo, par):
        """Computes and returns the skin temperature when thermally neutral.

        This method runs the JOS-3 model in uniform conditions at a temperature where PMV=0
        for the provided metabolic rate and clothing insulation values. If the given values match
        the cached ones, it directly returns the cached comfortable skin temperature set point.
        Otherwise, it calculates and returns a new comfortable skin temperature set point.

        Parameters
        ----------
        overall_clo : float
            Overall clothing insulation value [clo]
        par : float
            Physical activity ratio [-]

        Returns
        -------
        comfort_setpt_skin : np.array
            Skin temperature set point for sensation/comfort model

        Notes
        -----
        - If both `clo_overall` and `par` values match the cached values, this function will return
          the cached comfortable skin temperature set point directly.
        - The calculation of the comfortable skin temperature set point involves determining the operative temperature
          where PMV is zero and then running the model in uniform conditions to extract skin temperatures as set points.
        """
        # If the clothing insulation value 'clo' and the metabolic rate 'par' are the same as the cached values,
        # return the cached comfortable skin temperature set point
        if overall_clo == self.cached_overall_clo and par == self.cached_par:
            # print("Both 'clo' and 'par' are the same as the cached values. Cached values are applied to comfort set-points")
            return self.comfort_setpt_skin

        # Otherwise, calculate a new comfortable skin temperature set point
        else:
            # Calculate the metabolic rate based on the provided 'par' value.
            min_met = 1.0
            met = max(self.bmr * self.par / 58.15, min_met)

            # Calculate the metabolic rate based on the provided 'par' value.
            overall_clo = np.average(self._clo, weights=Default.local_bsa)

            # Cash PAR (physical activity ratio) and overall clothing insulation
            self.cached_par = self.par
            self.cached_overall_clo = self.overall_clo

            # Use the parent class's method to calculate the operative temperature where PMV is zero.
            self.to = self._calculate_operative_temp_when_pmv_is_zero(
                met=met, clo=overall_clo
            )

            # Run the model in uniform conditions at a temperature where PMV=0
            # for the given met and clo and use the resulting skin temperatures as the setpoints
            # Assuming that the initial conditions are uniform and at a temperature where PMV=0
            logging.debug("Skin neutral temperature calculation starts")
            logging.debug("Metabolic rate:", met)
            logging.debug("Overall clo value:", overall_clo)
            logging.debug("Operative temperature when PMV=0", self.to)

            for _ in range(10):
                self._run(dtime=600, passive=False)

            self.comfort_setpt_skin = self.t_skin
            logging.debug("Comfort set-point temperature:", self.comfort_setpt_skin)

            return self.comfort_setpt_skin

    def _calculate_skin_temperature_considering_sweating(
        self,
        t_skin,
        wet,
    ):
        """Calculate the skin temperature considering sweating for Zhang's model.

        This function takes into account the impact of sweating on the skin temperature. The Zhang model's
        prediction of local thermal sensation is represented by a logistic function of the difference between
        the current skin temperature and the neutral skin temperature. In a hot environment, sweating tends to
        suppress the increase in skin temperature, resulting in smaller differences in skin temperatures at
        locations with higher skin temperatures (like the chest or back) under neutral thermal conditions. This
        can lead to an underestimation of thermal sensations. Hence, this function modifies the skin temperature
        based on skin wettedness, utilizing a simple consideration.
        Based on the relationship between skin temperature and wettedness at isothermal sensation,
        skin temperature is corrected by increasing skin wettedness to improve prediction accuracy at warm environments.

        More details can be found in this Japanese research paper and data for TSV=3 is used as a reference:
        https://eprints.lib.hokudai.ac.jp/dspace/bitstream/2115/7826/1/4-2-6_p67-72.pdf

        Parameters
        ----------
        t_skin : float or numpy.ndarray
            Skin temperature [°C]
        wet : float
            skin wettedness [-]

        Returns
        -------
        t_skin_wet : float or numpy.ndarray
            Corrected skin temperature, considering latent heat [°C]

        Notes
        -----
        In the future, it may be more reasonable to predict thermal sensation
        based on the total heat loss (sensible + latent) of JOS3 model, as in the SET calculation method.
        """
        coef = 2.5
        t_skin_increase = coef * (wet - 0.06)
        t_skin_wet = t_skin + t_skin_increase
        return t_skin_wet

    def _run(self, *args, **kwargs):
        """Run the simulation for a single time step.

        This method extends the `_run` method of the parent class. In addition to the parent's operations, it
        calculates the rate of change of skin and core temperature and estimates thermal sensation and comfort parameters.

        Parameters
        ----------
        *args :
            Variable length argument list.
        **kwargs :
            Arbitrary keyword arguments.

        Returns
        -------
        dict_data : dict
            Dictionary containing data from the parent class `_run` method, rate of change of skin and core temperature,
            and other thermal sensation and comfort parameters.

        Notes
        -----
        - The skin temperature after considering sweat (`t_skin_wet`) is used for further calculations if `consider_sweat`
          is True. Otherwise, the regular skin temperature (`t_skin`) is used.
        - Local and overall thermal sensations and comfort are computed using the `cbe_comfort` function.
        """

        # Internal switch to consider sweat for comfort model
        consider_sweat = True

        # Define comfort set-point temperature
        comfort_setpt_skin = self._calculate_comfort_setpt(
            overall_clo=self.overall_clo, par=self.par
        )
        # Calculate the skin temperature after accounting for the effects of sweating.
        self.t_skin_wet = self._calculate_skin_temperature_considering_sweating(
            t_skin=self.t_skin, wet=self.w
        )

        # Use t_skin_wet for calculations if consider_sweat is True
        self.t_skin_used = self.t_skin_wet if consider_sweat else self.t_skin

        # Execute the `_run` method from the parent class and store its returned data.
        dict_data = super()._run(*args, **kwargs)

        # Retrieve or default the time difference value from keyword arguments.
        dtime = kwargs.get("dtime", 60)

        # Calculate the rate of change of skin and core temperatures over time.
        dt_skin = (
            (self.t_skin_used - self.t_skin_previous_time_step) / dtime
            if self.t_skin_previous_time_step is not None
            else 0
        )
        dt_core = (
            (self.t_core - self.t_core_previous_time_step) / dtime
            if self.t_core_previous_time_step is not None
            else 0
        )

        # Store the used skin and core temperatures for future calculations.
        self.t_skin_previous_time_step = self.t_skin_used
        self.t_core_previous_time_step = self.t_core

        # Update the dictionary with the rate of change values for skin and core temperatures.
        dict_data.update(
            {
                "dt_skin": dt_skin,
                "dt_core": dt_core,
            }
        )

        # Calculate thermal sensations and comfort parameters
        dict_sensation_and_comfort = zhang_sensation_comfort(
            t_skin_local=self.t_skin_used,
            dt_skin_local_dt=dt_skin,
            t_skin_local_set=comfort_setpt_skin,
            dt_core_local_dt=dt_core,
        )

        # Update the dictionary with the thermal sensation and comfort parameters
        dict_data.update(dict_sensation_and_comfort)

        return dict_data

    def simulate(self, *args, **kwargs):
        """Runs the model for a given time.
        This method extends the simulate method of the parent class by additionally computing the comfortable skin
        temperature set point before running the simulation.
        """
        result = super().simulate(*args, **kwargs)
        return result
