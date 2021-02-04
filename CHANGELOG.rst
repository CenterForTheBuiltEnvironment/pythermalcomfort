
Changelog
=========

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

* Changed default diameter in t_mrt
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
* Fixed error calculation of relative air velocity
* renamed input parameter ta to tdb
* Added function to calculate mean radiant temperature from black globe temperature
* Added function to calculate solar gain on people
* Added functions to calculate vapour pressure, wet-bulb temperature, dew point temperature, and psychrometric data from dry bulb temperature and RH
* Added authors
* Added dictionaries with reference clo and met values
* Added function to calculate enthalpy

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
