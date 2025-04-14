Changelog
=========

3.0.1 (2025-04-14)
-------------------

* allow np.float and np.int as inputs to all functions
* fixed documentation for phs - met units

3.0.0 (2025-02-03)
-------------------

.. warning::
    pythermalcomfort version 3.0.0 introduces some breaking changes.

    **How functions return results:**
    as the functions now return dataclass instances with the calculation results.
    This change enhances the structure and accessibility of the results.
    For example:

    .. code-block:: python

        from pythermalcomfort.models import pmv_ppd_iso
        result = pmv_ppd_iso(tdb=[22, 25], tr=25, vr=0.1, rh=50, met=1.4, clo=0.5, model='7730-2005')
        print(result.pmv)  # [-0.  0.41]

    This update aims to make the package more user-friendly and to provide a more organized way to access all calculation results.

    **Moved functions**
    Moved all the functions that were in the `psychrometrics.py` file to the `utilities.py` file.

    **Changed function names**
    All the PMV functions have been renamed using the following format: `pmv_XXX` where XXX is the standard or the model name.

    **PMV function**
    The pmv_ppd function now has been split into two functions: pmv_ppd_iso and pmv_ppd_ashrae.

.. note::
    We have updated all functions to accept Numpy arrays as inputs, allowing you to pass multiple values at once for faster results.
    Single values are still accepted, and the functions will return results as before.
    Additionally, we have synchronized the tests with the R comf package to ensure consistent calculation results across both packages.

    Other improvements include:

    * Enhanced documentation with more examples.
    * Better described the models.
    * Added more tests to ensure calculation accuracy.
    * Implemented input validation to ensure inputs are within model applicability limits.
    * Harmonized input names across all functions.
    * Added surveys to assess thermal comfort to the documentation.
    * Added a detailed section about clothing insulation.

2.10.0 (2024-03-18)
-------------------

* allow n-dimensional arrays for ``pet_steady`` and speedup ``p_sat`` calculation


2.9.1 (2024-01-19)
-------------------

* Fixed error calculation of mass sweating in PET mode, the unit was incorrect

2.9.0 (2024-01-15)
-------------------

.. warning::
    pythermalcomfort 2.9.0 is no longer compatible with Python 3.8

* The PHS model accepts arrays as inputs

2.8.11 (2023-10-26)
-------------------

* wrote more test and improved code

2.8.11 (2023-10-26)
-------------------

* fixed issues with the documentation and sorted the models in alphabetical order

2.8.7 (2023-10-23)
-------------------

* Adaptive ASHRAE now returns a dataclass

2.8.6 (2023-10-09)
-------------------

* re-structured and linted the code

2.8.4 (2023-09-20)
-------------------

* calculation of cooling effect in pmv (standard='ashrae') triggered only when v>0.1 m/s

2.8.3 (2023-09-14)
-------------------

* general improvements in the JOS3 model

2.8.2 (2023-09-04)
-------------------

* general improvements in the JOS3 model
* fixed error when e_max == 0

2.8.1 (2023-07-05)
-------------------

* pythermalcomfort needs Python version > 3.8
* fixed issue in Cooling Effect calculation

2.8.0 (2023-07-03)
-------------------

* allowing the cooling effect to range from 0 to 40
* fixed PHS documentation
* improved JOS3 documentation

2.7.0 (2023-02-16)
-------------------

* changed coefficient of vasodilation in set_tmp() to 120 to match ASHRAE 55 2020 code
* slightly modified value in validation tables

2.6.0 (2023-01-17)
-------------------

* max sweating rate can be passed to two node model
* max skin wettedness can be passed to two node model
* rounding w to two decimals
* use_fans_heatwave function accepts arrays
* fixed typos unit documentation

2.5.4 (2022-10-12)
-------------------

* PHS model accepts all required inputs to be run on a minute by minute basis
* fix error check compliance PHS model

2.5.0 (2022-06-13)
-------------------

* Added the adaptive thermal heat balance (ATHB) model

2.4.0 (2022-06-10)
-------------------

* Added e_pmv model - Adjusted Predicted Mean Votes with Expectancy Factor
* Added a_pmv model - Adaptive Predicted Mean Vote

2.3.0 (2022-06-01)
-------------------

* Added discomfort index

2.2.0 (2022-05-17)
-------------------

* Implemented a better equation to calculate the mean radiant temperature

2.1.1 (2022-05-17)
-------------------

* Fixed how DISC is calculated

2.1.0 (2022-04-20)
-------------------

* Added Physiological Equivalent Temperature (PET) model
* In PMV and PPD function you can specify if occupants has control over airspeed

2.0.2 (2022-04-12)
-------------------

* UTCI accepts lists as inputs

2.0.0 (2022-04-07)
-------------------

.. warning::
    Version 2.0.0 introduces some breaking changes. Now the default behaviour of most of the function is that they return a ``np.nan`` if the inputs are outside the model applicability limits.

    For most functions we are no longer printing ``Warnings``. If you want the function to return a value even if your inputs are outside the model applicability limits then you can set the variable ``limit_input = False``. Please note that you should refrain from doing this.


.. note::
    Starting from Version 2.0.0 of pythermalcomfort now most of the functions (see detailed list below) accept Numpy arrays or lists as inputs. This allows you to write more concise and faster code since we optimized vectorization, where possible using Numba.

* Allowing users to pass Numpy arrays or lists as input to the pmv_ppd, pmv, clo_tout, both adaptive models, utci, set_tmp, two_nodes
* Changed the input variable from return_invalid to limit_input
* Increased speed by using Numba @vectorize decorator
* Changed ASHRAE 55 2020 limits to match new addenda
* Improved documentation

1.11.0 (2022-03-16)
-------------------

* Allowing users to pass a Numpy array as input into the UTCI function
* Numpy is now a requirement of pythermalcomfort
* Improved PMV, JOS-3, and UTCI documentation
* Testing PMV, SET, and solar gains models using online reference tables

1.10.0 (2021-11-15)
-------------------

* Added JOS-3 model

1.9.0 (2021-10-07)
------------------

* Added Normal Effective Temperature (NET)
* Added Apparent Temperature (AT)
* Added Wind Chill Index (WCI)

1.8.0 (2021-09-28)
------------------

* Gagge's two-node model
* Added WBGT equation
* Added Heat index (HI)
* Added humidex index

1.7.1 (2021-09-08)
------------------

* Added ASHRAE equation to calculate the operative temperature

1.7.0 (2021-07-29)
------------------

* Implemented function to calculate the if fans are beneficial during heatwaves
* Fixed error in the SET equation to calculated radiative heat transfer coefficient
* Fixed error in SET definition
* Moved functions optimized with Numba to new file

1.6.2 (2021-07-08)
------------------

* Updated equation clo_dynamic based on ANSI/ASHRAE Addendum f to ANSI/ASHRAE Standard 55-2020
* Fixed import errors in examples

1.6.1 (2021-07-05)
------------------

* optimized UTCI function with Numba

1.6.0 (2021-05-21)
------------------

* (BREAKING CHANGE) moved some of the functions from psychrometrics to utilities
* added equation to calculate body surface area

1.5.2 (2021-05-05)
------------------

* return stress category UTCI

1.5.1 (2021-04-29)
------------------

* optimized phs with Numba

1.5.0 (2021-04-21)
------------------

* added Predicted Heat Strain (PHS) index from ISO 7933:2004

1.4.6 (2021-03-30)
------------------

* changed equation to calculate convective heat transfer coefficient in set_tmp() as per Gagge's 1986
* fixed vasodilation coefficient in set_tmp()
* docs changed term air velocity with air speed and improved documentation
* added new tests for comfort functions

1.3.6 (2021-02-04)
------------------

* fixed error calculation solar_altitude and sharp for supine person in solar_gain

1.3.5 (2021-02-02)
------------------

* not rounding SET temperature when calculating cooling effect

1.3.3 (2020-12-14)
------------------

* added function to calculate sky-vault view fraction

1.3.2 (2020-12-14)
------------------

* replaced input solar_azimuth with sharp in the solar_gain() function
* fixed small error in example pmv calculation

1.3.1 (2020-10-30)
------------------

* Fixed error calculation of cooling effect with elevated air temperatures

1.3.0 (2020-10-19)
------------------

* Changed PMV elevated air speed limit from 0.2 to 0.1 m/s

1.2.3 (2020-09-09)
------------------

* Fixed error in the calculation of erf
* Updated validation table erf

1.2.2 (2020-08-21)
------------------

* Changed default diameter in mean_radiant_tmp
* Improved documentation


1.2.0 (2020-07-29)
------------------

* Significantly improved calculation speed using numba. Wrapped set and pmv functions

1.0.6 (2020-07-24)
------------------

* Minor speed improvement changed math.pow with **
* Added validation PMV validation table from ISO 7730

1.0.4 (2020-07-20)
------------------

* Improved speed calculation of the Cooling Effect
* Bisection has been replaced with Brentq function from scipy

1.0.3 (2020-07-01)
------------------

* Annotated variables in the SET code.

1.0.2 (2020-06-11)
------------------

* Fixed an error in the bisection equation used to calculated Cooling Effect.


1.0.0 (2020-06-09)
------------------

* Major stable release.

0.7.0 (2020-06-09)
------------------

* Added equation to calculate the dynamic clothing insulation

0.6.3 (2020-04-11)
------------------

* Fixed error in calculation adaptive ASHRAE
* Added some examples

0.6.3 (2020-03-17)
------------------

* Renamed function to_calc to t_o
* Fixed error calculation of relative air speed
* renamed input parameter ta to tdb
* Added function to calculate mean radiant temperature from black globe temperature
* Added function to calculate solar gain on people
* Added functions to calculate vapour pressure, wet-bulb temperature, dew point temperature, and psychrometric data from dry bulb temperature and RH
* Added authors
* Added dictionaries with reference clo and met values
* Added function to calculate enthalpy_air

0.5.2 (2020-03-11)
------------------

* Added function to calculate the running mean outdoor temperature

0.5.1 (2020-03-06)
------------------

* There was an error in version 0.4.2 in the calculation of PMV and PPD with elevated air speed, i.e. vr > 0.2 which has been fixed in this version
* Added function to calculate the cooling effect in accordance with ASHRAE

0.4.1 (2020-02-17)
------------------

* Removed compatibility with python 2.7 and 3.5

0.4.0 (2020-02-17)
------------------

* Created adaptive_EN, v_relative, t_clo, vertical_tmp_gradient, ankle_draft functions and wrote tests.
* Added possibility to decide with measuring system to use SI or IP.

0.3.0 (2020-02-13)
------------------

* Created set_tmp, adaptive_ashrae, UTCI functions and wrote tests.
* Added warning to let the user know if inputs entered do not comply with Standards applicability limits.

0.1.0 (2020-02-11)
------------------

* Created pmv, pmv_ppd functions and wrote tests.
* Documented code.

0.0.0 (2020-02-11)
------------------

* First release on PyPI.
