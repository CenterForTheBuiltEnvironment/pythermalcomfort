Comfort models
==============

.. testsetup::

    from pythermalcomfort import *
    from pythermalcomfort.models import *


PMV PPD
-------

.. autofunction:: pythermalcomfort.models.pmv_ppd

PMV
---

.. autofunction:: PYTHERMALCOMFORT.models.pmv

Standard Effective Temperature (SET)
------------------------------------

.. autofunction:: pythermalcomfort.models.set_tmp

Cooling Effect
--------------

.. autofunction:: pythermalcomfort.models.cooling_effect

Adaptive ASHRAE
---------------

.. autofunction:: pythermalcomfort.models.adaptive_ashrae

Adaptive EN
-----------

.. autofunction:: pythermalcomfort.models.adaptive_en

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


**References**

.. [1] ANSI, & ASHRAE. (2017). Thermal Environmental Conditions for Human Occupancy. Atlanta.
.. [2] ISO. (2005). ISO 7730 - Ergonomics of the thermal environment — Analytical determination and interpretation of thermal comfort using calculation of the PMV and PPD indices and local thermal comfort criteria.
.. [3] EN, & BSI. (2019). Energy performance of buildings - Ventilation for buildings. BSI Standards Limited 2019.
.. [4] Schiavon, S., & Lee, K. H. (2013). Dynamic predictive clothing insulation models based on outdoor air and indoor operative temperatures. Building and Environment, 59, 250–260. doi.org/10.1016/j.buildenv.2012.08.024
.. [5] ISO. (1998). ISO 7726 - Ergonomics of the thermal environment instruments for measuring physical quantities.
.. [6] Stull, R., 2011. Wet-Bulb Temperature from Relative Humidity and Air Temperature. J. Appl. Meteorol. Climatol. 50, 2267–2269. doi.org/10.1175/JAMC-D-11-0143.1
.. [7] Zare, S., Hasheminejad, N., Shirvan, H.E., Hemmatjo, R., Sarebanzadeh, K., Ahmadi, S., 2018. Comparing Universal Thermal Climate Index (UTCI) with selected thermal indices/environmental parameters during 12 months of the year. Weather Clim. Extrem. 19, 49–57. https://doi.org/10.1016/j.wace.2018.01.004

Psychrometrics functions
========================

.. automodule:: pythermalcomfort.psychrometrics
    :members:

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

