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
      - |travis| |codecov|
    * - package
      - | |version| |wheel|
        | |supported-ver|
        | |package-health|

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

.. |travis| image:: https://api.travis-ci.org/CenterForTheBuiltEnvironment/pythermalcomfort.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/CenterForTheBuiltEnvironment/pythermalcomfort

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

Please cite us if you use this package: `Tartarini, F., Schiavon, S., 2020. pythermalcomfort: A Python package for thermal comfort research. SoftwareX 12, 100578. https://doi.org/10.1016/j.softx.2020.100578 <https://www.sciencedirect.com/science/article/pii/S2352711020302910>`_

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

.. _Examples: https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort/tree/master/examples

YouTube `tutorials`_ playlist

.. _tutorials: https://www.youtube.com/playlist?list=PLY91jl6VVD7zMaJjRVrVkaBtI56U7ztQC


Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit helps, and credit will always be given. Click `here`_  to learn more on how to contribute to the project.

.. _here: https://pythermalcomfort.readthedocs.io/en/latest/contributing.html


Deployment
==========

I am using travis to test the code. In addition, I have enabled GitHub actions. Every time a commit containing the message `bump version` is pushed to master then the GitHub action tests the code and if the tests pass, a new version of the package is published automatically on PyPI. See file in `.github/workflows/` for more information.

