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

.. autofunction:: pythermalcomfort.models.pmv

Standard Effective Temperature
------------------------------

.. autofunction:: pythermalcomfort.models.set_tmp

Adaptive ASHRAE
---------------

.. autofunction:: pythermalcomfort.models.adaptive_ashrae

Adaptive EN
-----------

.. autofunction:: pythermalcomfort.models.adaptive_en

Universal Thermal Climate Index (UTCI)
--------------------------------------

.. autofunction:: pythermalcomfort.models.utci

Clothing prediction
-------------------

.. autofunction:: pythermalcomfort.models.clo_tout

Vertical air temperature gradient
---------------------------------

.. autofunction:: pythermalcomfort.models.vertical_tmp_grad_ppd


**References**

.. [1] ANSI, & ASHRAE. (2017). Thermal Environmental Conditions for Human Occupancy. Atlanta.
.. [2] ISO. (2005). ISO 7730 - Ergonomics of the thermal environment — Analytical determination and interpretation of thermal comfort using calculation of the PMV and PPD indices and local thermal comfort criteria.
.. [3] EN, & BSI. (2019). Energy performance of buildings - Ventilation for buildings. BSI Standards Limited 2019.
.. [4] Schiavon, S., & Lee, K. H. (2013). Dynamic predictive clothing insulation models based on outdoor air and indoor operative temperatures. Building and Environment, 59, 250–260. https://doi.org/10.1016/j.buildenv.2012.08.024
.. [5] ISO. (1998). ISO 7726 - Ergonomics of the thermal environment instruments for measuring physical quantities.

Psychrometrics functions
========================

.. automodule:: pythermalcomfort.psychrometrics
    :members:
