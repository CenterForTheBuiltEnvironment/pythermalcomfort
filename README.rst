========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - license
      - |license|
    * - downloads
      - |downloads|
    * - tests
      - | |appveyor|
        | |codecov|
        | |tests|
    * - package
      - | |version| |wheel|
        | |supported-ver|
        | |package-health|

.. |tests| image:: https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort/actions/workflows/build-test-publish.yml/badge.svg
    :target: https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort/actions/workflows/build-test-publish.yml
    :alt: Tests to make sure pythermalcomfort works on different Python versions and OS

.. |package-health| image:: https://snyk.io/advisor/python/pythermalcomfort/badge.svg
    :target: https://snyk.io/advisor/python/pythermalcomfort
    :alt: pythermalcomfort

.. |license| image:: https://img.shields.io/pypi/l/pythermalcomfort?color=brightgreen
    :target: https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort/blob/master/LICENSE
    :alt: pythermalcomfort license

.. |docs| image:: https://readthedocs.org/projects/pythermalcomfort/badge/?style=flat
    :target: https://readthedocs.org/projects/pythermalcomfort
    :alt: Documentation Status

.. |downloads| image:: https://img.shields.io/pypi/dm/pythermalcomfort?color=brightgreen
    :alt: PyPI - Downloads

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/CenterForTheBuiltEnvironment/pythermalcomfort?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/CenterForTheBuiltEnvironment/pythermalcomfort

.. |codecov| image:: https://codecov.io/github/CenterForTheBuiltEnvironment/pythermalcomfort/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/CenterForTheBuiltEnvironment/pythermalcomfort

.. |version| image:: https://img.shields.io/pypi/v/pythermalcomfort.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/pythermalcomfort

.. |wheel| image:: https://img.shields.io/pypi/wheel/pythermalcomfort.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/pythermalcomfort

.. |supported-ver| image:: https://img.shields.io/pypi/pyversions/pythermalcomfort.svg
    :alt: Supported versions
    :target: https://pypi.org/project/pythermalcomfort

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/pythermalcomfort.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/pythermalcomfort

.. end-badges

Package to calculate several thermal comfort indices (e.g. PMV, PPD, SET, adaptive) and convert physical variables.

Please cite us if you use this package: `Tartarini, F., Schiavon, S., 2020. pythermalcomfort: A Python package for thermal comfort research. SoftwareX 12, 100578. https://doi.org/10.1016/j.softx.2020.100578 <https://doi.org/10.1016/j.softx.2020.100578>`_

* Free software: MIT license

Installation
============

::

    pip install pythermalcomfort

You can also install the in-development version with::

    pip install https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort/archive/master.zip


Documentation
=============


https://pythermalcomfort.readthedocs.io/


Examples and Tutorials
======================

`Examples`_ files on how to use some of the functions

.. _Examples: https://pythermalcomfort.readthedocs.io/en/latest/usage.html


Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given. Click `here`_  to learn more on how to contribute to the project.

.. _here: https://pythermalcomfort.readthedocs.io/en/latest/contributing.html


Deployment
==========

I am using travis to test the code. In addition, I have enabled GitHub actions.
Every time the code is pushed or pulled to the `master` repository then the GitHub action tests the code and if the tests pass, a new version of the package is published automatically on PyPI.
See file in `.github/workflows/` for more information.


Test (For developers)
=====================
To run all the test cases with scripts_test_result.py script after any changes. The test cases come from `validation-data-comfort-models`_

.. _validation-data-comfort-models: https://github.com/FedericoTartarini/validation-data-comfort-models

    ```bash
    python ./tests/scripts_test_result.py [env]
    ```
`env` is used to specify the Python version of test environment. It could be py39, py310, py311, py312. The scripts will call tox command and summarize test results into a markdown table. 
The following markdown table will be updated once you run the scriput.

----------------------------
| Test File                     | Result   |
|:------------------------------|:---------|
| test_a_pmv.py                 | PASSED   |
| test_adaptive_ashrae.py       | PASSED   |
| test_adaptive_en.py           | PASSED   |
| test_ankle_draft.py           | PASSED   |
| test_at.py                    | PASSED   |
| test_athb.py                  | PASSED   |
| test_clo_tout.py              | PASSED   |
| test_cooling_effect.py        | PASSED   |
| test_discomfort_index.py      | PASSED   |
| test_e_pmv.py                 | PASSED   |
| test_heat_index.py            | PASSED   |
| test_humidex.py               | PASSED   |
| test_jos3.py                  | PASSED   |
| test_net.py                   | PASSED   |
| test_pet_steady.py            | PASSED   |
| test_phs.py                   | PASSED   |
| test_pmv.py                   | PASSED   |
| test_pmv_ppd.py               | PASSED   |
| test_psychrometrics.py        | PASSED   |
| test_set.py                   | PASSED   |
| test_solar_gain.py            | PASSED   |
| test_two_nodes.py             | PASSED   |
| test_use_fans_heatwaves.py    | PASSED   |
| test_utci.py                  | PASSED   |
| test_utilities.py             | PASSED   |
| test_vertical_tmp_grad_ppd.py | PASSED   |
| test_wbgt.py                  | PASSED   |
| test_wind_chill.py            | PASSED   |
----------------------------
