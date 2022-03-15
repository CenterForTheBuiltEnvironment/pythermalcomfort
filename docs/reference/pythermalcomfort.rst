Comfort models
==============

.. testsetup::

    from pythermalcomfort import *
    from pythermalcomfort.models import *


Predicted Mean Vote (PMV) and Predicted Percentage of Dissatisfied (PPD)
------------------------------------------------------------------------

.. autofunction:: pythermalcomfort.models.pmv_ppd

Predicted Mean Vote (PMV)
-------------------------

.. autofunction:: pythermalcomfort.models.pmv

Gagge et al. two-node model
---------------------------

.. autofunction:: pythermalcomfort.models.two_nodes

Standard Effective Temperature (SET)
------------------------------------

.. autofunction:: pythermalcomfort.models.set_tmp

Cooling Effect (CE)
-------------------

.. autofunction:: pythermalcomfort.models.cooling_effect

Joint system thermoregulation model (JOS-3)
-------------------------------------------

    JOS-3 is a numeric model to simulate a human thermoregulation [19]_.
    The JOS-3 model consists of 83 nodes. Human physiological responses and body temperatures are calculated using the backward difference method. JOS-3 uses brown adipose tissue activity, aging effects, and heat gain by shortwave solar radiation at the skin to predict human physiological responses. It also considers personal characteristics in transient and non-uniform thermal environments. The JOS-3 was validated by comparing the results with those of human subject tests conducted under stable and transient conditions [19]_.

    To read the JOS-3 official documentation please use the following commands:

    .. code-block:: python

        >>> import jos3
        >>> model = jos3.JOS3()

        >>> # Print documentation:
        >>> print(model.__doc__)

        >>> # Show the documentation of the output parameters:
        >>> print(jos3.show_outparam_docs())


    Below an example on how to use the JOS-3 model

    .. code-block:: python

        >>> import pandas as pd
        >>> import jos3

        >>> model = jos3.JOS3(height=1.7, weight=60, age=30)  # Builds a model

        >>> # Set the first condition
        >>> model.To = 28  # Operative temperature [oC]
        >>> model.RH = 40  # Relative humidity [%]
        >>> model.Va = 0.2  # Air velocity [m/s]
        >>> model.PAR = 1.2  # Physical activity ratio [-]
        >>> model.simulate(60)  # Exposure time = 60 [min]

        >>> # Set the next condition
        >>> model.To = 20  # Changes only operative temperature
        >>> model.simulate(60)  # Additional exposure time = 60 [min]

        >>> # Show the results
        >>> df = pd.DataFrame(model.dict_results())  # Make pandas.DataFrame
        >>> df.TskMean.plot()  # Show the graph of mean skin temp.


Adaptive ASHRAE
---------------

.. autofunction:: pythermalcomfort.models.adaptive_ashrae

Adaptive EN
-----------

.. autofunction:: pythermalcomfort.models.adaptive_en

Use Fans During Heatwaves
-------------------------

.. autofunction:: pythermalcomfort.models.use_fans_heatwaves

Solar gain on people
--------------------

.. autofunction:: pythermalcomfort.models.solar_gain

Universal Thermal Climate Index (UTCI)
--------------------------------------

.. autofunction:: pythermalcomfort.models.utci

Clothing prediction
-------------------

.. autofunction:: pythermalcomfort.models.clo_tout

Vertical air temperature gradient
---------------------------------

.. autofunction:: pythermalcomfort.models.vertical_tmp_grad_ppd

Ankle draft
-----------

.. autofunction:: pythermalcomfort.models.ankle_draft

Predicted Heat Strain (PHS) Index
---------------------------------

.. autofunction:: pythermalcomfort.models.phs

Wet Bulb Globe Temperature Index (WBGT)
---------------------------------------

.. autofunction:: pythermalcomfort.models.wbgt

Heat Index (HI)
---------------

.. autofunction:: pythermalcomfort.models.heat_index

Humidex
-------

.. autofunction:: pythermalcomfort.models.humidex

Normal Effective Temperature (NET)
----------------------------------

.. autofunction:: pythermalcomfort.models.net

Wind chill index
----------------

.. autofunction:: pythermalcomfort.models.wc

Apparent Temperature (AT)
-------------------------

.. autofunction:: pythermalcomfort.models.at

Psychrometrics functions
========================

.. automodule:: pythermalcomfort.psychrometrics
    :members:

Utilities functions
===================

Body Surface Area
-----------------

.. autofunction:: pythermalcomfort.utilities.body_surface_area

Relative air speed
------------------

.. autofunction:: pythermalcomfort.utilities.v_relative

Dynamic clothing
----------------

.. autofunction:: pythermalcomfort.utilities.clo_dynamic

Running mean outdoor temperature
--------------------------------

.. autofunction:: pythermalcomfort.utilities.running_mean_outdoor_temperature

Units converter
---------------

.. autofunction:: pythermalcomfort.utilities.units_converter

Sky-vault view fraction
-----------------------

.. autofunction:: pythermalcomfort.utilities.f_svv

Reference values clo and met
============================

Met typical tasks, [met]
------------------------

.. autodata:: pythermalcomfort.utilities.met_typical_tasks

**Example**

.. code-block:: python

    >>> from pythermalcomfort.utilities import met_typical_tasks
    >>> print(met_typical_tasks['Filing, standing'])
    1.4

Clothing insulation of typical ensembles, [clo]
-----------------------------------------------

.. autodata:: pythermalcomfort.utilities.clo_typical_ensembles

**Example**

.. code-block:: python

    >>> from pythermalcomfort.utilities import clo_typical_ensembles
    >>> print(clo_typical_ensembles['Typical summer indoor clothing'])
    0.5

Insulation of individual garments, [clo]
----------------------------------------

.. autodata:: pythermalcomfort.utilities.clo_individual_garments

**Example**

.. code-block:: python

    >>> from pythermalcomfort.utilities import clo_individual_garments
    >>> print(clo_individual_garments['T-shirt'])
    0.08

    >>> # calculate total clothing insulation
    >>> i_cl = clo_individual_garments['T-shirt'] + clo_individual_garments["Men's underwear"] +
    >>>        clo_individual_garments['Thin trousers'] + clo_individual_garments['Shoes or sandals']
    >>> print(i_cl)
    0.29

**References**

.. [1] ANSI, & ASHRAE. (2020). Thermal Environmental Conditions for Human Occupancy. Atlanta.
.. [2] ISO. (2005). ISO 7730 - Ergonomics of the thermal environment — Analytical determination and interpretation of thermal comfort using calculation of the PMV and PPD indices and local thermal comfort criteria.
.. [3] EN, & BSI. (2019). Energy performance of buildings - Ventilation for buildings. BSI Standards Limited 2019.
.. [4] Schiavon, S., & Lee, K. H. (2013). Dynamic predictive clothing insulation models based on outdoor air and indoor operative temperatures. Building and Environment, 59, 250–260. doi.org/10.1016/j.buildenv.2012.08.024
.. [5] ISO. (1998). ISO 7726 - Ergonomics of the thermal environment instruments for measuring physical quantities.
.. [6] Stull, R., 2011. Wet-Bulb Temperature from Relative Humidity and Air Temperature. J. Appl. Meteorol. Climatol. 50, 2267–2269. doi.org/10.1175/JAMC-D-11-0143.1
.. [7] Zare, S., Hasheminejad, N., Shirvan, H.E., Hemmatjo, R., Sarebanzadeh, K., Ahmadi, S., 2018. Comparing Universal Thermal Climate Index (UTCI) with selected thermal indices/environmental parameters during 12 months of the year. Weather Clim. Extrem. 19, 49–57. https://doi.org/10.1016/j.wace.2018.01.004
.. [8] ISO, 2004. ISO 7933 - Ergonomics of the thermal environment — Analytical determination and interpretation of heat stress using calculation of the predicted heat strain.
.. [9] Błażejczyk, K., Jendritzky, G., Bröde, P., Fiala, D., Havenith, G., Epstein, Y., Psikuta, A. and Kampmann, B., 2013. An introduction to the universal thermal climate index (UTCI). Geographia Polonica, 86(1), pp.5-10.
.. [10] Gagge, A.P., Fobelets, A.P., and Berglund, L.G., 1986. A standard predictive Index of human reponse to thermal enviroment. Am. Soc. Heating, Refrig. Air-Conditioning Eng. 709–731.
.. [11] ISO, 2017. ISO 7243 - Ergonomics of the thermal environment — Assessment of heat stress using the WBGT (wet bulb globe temperature) index.
.. [12] Rothfusz LP (1990) The heat index equation. NWS Southern Region Technical Attachment, SR/SSD 90–23, Fort Worth, Texas
.. [13] Steadman RG (1979) The assessment of sultriness. Part I: A temperature-humidity index based on human physiology and clothing science. J Appl Meteorol 18:861–873
.. [14] Masterton JM, Richardson FA. Humidex, a method of quantifying human discomfort due to excessive heat and humidity. Downsview, Ontario: CLI 1-79, Environment Canada, Atmospheric Environment Service, 1979
.. [15] Havenith, G., Fiala, D., 2016. Thermal indices and thermophysiological modeling for heat stress. Compr. Physiol. 6, 255–302. https://doi.org/10.1002/cphy.c140051
.. [16] Blazejczyk, K., Epstein, Y., Jendritzky, G., Staiger, H., Tinz, B., 2012. Comparison of UTCI to selected thermal indices. Int. J. Biometeorol. 56, 515–535. https://doi.org/10.1007/s00484-011-0453-2
.. [17] Steadman RG (1984) A universal scale of apparent temperature. J Appl Meteorol Climatol 23:1674–1687
.. [18] ASHRAE, 2017. 2017 ASHRAE Handbook Fundamentals. Atlanta.
.. [19] Takahashi, Y., Nomoto, A., Yoda, S., Hisayama, R., Ogata, M., Ozeki, Y., & Tanabe, S. ichi. (2021). Thermoregulation model JOS-3 with new open source code. Energy and Buildings, 231, 110575. https://doi.org/10.1016/j.enbuild.2020.110575

