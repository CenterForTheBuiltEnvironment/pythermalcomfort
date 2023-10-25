Comfort models
==============

Adaptive ASHRAE
---------------

.. autofunction:: pythermalcomfort.models.adaptive_ashrae.adaptive_ashrae

Adaptive EN
-----------

.. autofunction:: pythermalcomfort.models.adaptive_en.adaptive_en

Adaptive Predicted Mean Vote (aPMV)
-----------------------------------

.. autofunction:: pythermalcomfort.models.a_pmv.a_pmv

Adaptive Thermal Heat Balance (ATHB)
------------------------------------

.. autofunction:: pythermalcomfort.models.athb.athb

Adjusted Predicted Mean Votes with Expectancy Factor (ePMV)
-----------------------------------------------------------

.. autofunction:: pythermalcomfort.models.e_pmv.e_pmv

Apparent Temperature (AT)
-------------------------

.. autofunction:: pythermalcomfort.models.at.at

Ankle draft
-----------

.. autofunction:: pythermalcomfort.models.ankle_draft.ankle_draft

Clothing prediction
-------------------

.. autofunction:: pythermalcomfort.models.clo_tout.clo_tout

Cooling Effect (CE)
-------------------

.. autofunction:: pythermalcomfort.models.cooling_effect.cooling_effect

Discomfort Index (DI)
---------------------

.. autofunction:: pythermalcomfort.models.discomfort_index.discomfort_index

Heat Index (HI)
---------------

.. autofunction:: pythermalcomfort.models.heat_index.heat_index

Humidex
-------

.. autofunction:: pythermalcomfort.models.humidex.humidex

Joint system thermoregulation model (JOS-3)
-------------------------------------------

.. autoclass:: pythermalcomfort.models.jos3.JOS3
    :members:
    :undoc-members:
    :special-members: __init__

Normal Effective Temperature (NET)
----------------------------------

.. autofunction:: pythermalcomfort.models.net.net

Predicted Heat Strain (PHS) Index
---------------------------------

.. autofunction:: pythermalcomfort.models.phs.phs

Physiological Equivalent Temperature (PET)
------------------------------------------

.. autofunction:: pythermalcomfort.models.pet_steady.pet_steady

Predicted Mean Vote (PMV) and Predicted Percentage of Dissatisfied (PPD)
------------------------------------------------------------------------

.. autofunction:: pythermalcomfort.models.pmv_ppd.pmv_ppd

Predicted Mean Vote (PMV)
-------------------------

.. autofunction:: pythermalcomfort.models.pmv.pmv

Solar gain on people
--------------------

.. autofunction:: pythermalcomfort.models.solar_gain.solar_gain

Standard Effective Temperature (SET)
------------------------------------

.. autofunction:: pythermalcomfort.models.set_tmp.set_tmp

Two-node model
--------------

.. autofunction:: pythermalcomfort.models.two_nodes.two_nodes

Universal Thermal Climate Index (UTCI)
--------------------------------------

.. autofunction:: pythermalcomfort.models.utci.utci

Use Fans During Heatwaves
-------------------------

.. autofunction:: pythermalcomfort.models.use_fans_heatwaves.use_fans_heatwaves

Vertical air temperature gradient
---------------------------------

.. autofunction:: pythermalcomfort.models.vertical_tmp_grad_ppd.vertical_tmp_grad_ppd

Wet Bulb Globe Temperature Index (WBGT)
---------------------------------------

.. autofunction:: pythermalcomfort.models.wbgt.wbgt

Wind chill index
----------------

.. autofunction:: pythermalcomfort.models.wc.wc

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
.. [15] Havenith, G., Fiala, D., 2016. Thermal indices and thermophysiological modeling for heat stress. Compr. Physiol. 6, 255–302. DOI: doi.org/10.1002/cphy.c140051
.. [16] Blazejczyk, K., Epstein, Y., Jendritzky, G., Staiger, H., Tinz, B., 2012. Comparison of UTCI to selected thermal indices. Int. J. Biometeorol. 56, 515–535. DOI: doi.org/10.1007/s00484-011-0453-2
.. [17] Steadman RG (1984) A universal scale of apparent temperature. J Appl Meteorol Climatol 23:1674–1687
.. [18] ASHRAE, 2017. 2017 ASHRAE Handbook Fundamentals. Atlanta.
.. [20] Höppe P. The physiological equivalent temperature - a universal index for the biometeorological assessment of the thermal environment. Int J Biometeorol. 1999 Oct;43(2):71-5. doi: 10.1007/s004840050118. PMID: 10552310.
.. [21] Walther, E. and Goestchel, Q., 2018. The PET comfort index: Questioning the model. Building and Environment, 137, pp.1-10. DOI: doi.org/10.1016/j.buildenv.2018.03.054
.. [22] Teitelbaum, E., Alsaad, H., Aviv, D., Kim, A., Voelker, C., Meggers, F., & Pantelic, J. (2022). Addressing a systematic error correcting for free and mixed convection when measuring mean radiant temperature with globe thermometers. Scientific Reports, 12(1), 1–18. DOI: doi.org/10.1038/s41598-022-10172-5
.. [23] Liu, S., Schiavon, S., Kabanshi, A., Nazaroff, W.W., 2017. Predicted percentage dissatisfied with ankle draft. Indoor Air 27, 852–862. DOI: doi.org/10.1111/ina.12364
.. [24] Polydoros, Anastasios & Cartalis, Constantinos. (2015). Use of Earth Observation based indices for the monitoring of built-up area features and dynamics in support of urban energy studies. Energy and Buildings. 98. 92-99. 10.1016/j.enbuild.2014.09.060.
.. [25] Yao, Runming & Li, Baizhan & Liu, Jing. (2009). A theoretical adaptive model of thermal comfort – Adaptive Predicted Mean Vote (aPMV). Building and Environment. 44. 2089-2096. 10.1016/j.buildenv.2009.02.014.
.. [26] Fanger, P. & Toftum, Jorn. (2002). Extension of the PMV model to non-air-conditioned buildings in warm climates. Energy and Buildings. 34. 533-536. 10.1016/S0378-7788(02)00003-8.
.. [27] Schweiker, M., 2022. Combining adaptive and heat balance models for thermal sensation prediction: A new approach towards a theory and data‐driven adaptive thermal heat balance model. Indoor Air 32, 1–19. DOI: doi.org/10.1111/ina.13018

