========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor|
        | |codecov| |requires|
    * - package
      - | |version| |wheel|
        | |supported-versions|
        | |supported-implementations|

.. |docs| image:: https://readthedocs.org/projects/pythermalcomfort/badge/?style=flat
    :target: https://readthedocs.org/projects/pythermalcomfort
    :alt: Documentation Status

.. |travis| image:: https://api.travis-ci.org/CenterForTheBuiltEnvironment/pythermalcomfort.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/CenterForTheBuiltEnvironment/pythermalcomfort

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/CenterForTheBuiltEnvironment/pythermalcomfort?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/CenterForTheBuiltEnvironment/pythermalcomfort

.. |requires| image:: https://requires.io/github/CenterForTheBuiltEnvironment/pythermalcomfort/requirements.svg?branch=master
    :alt: Requirements Status
    :target: https://requires.io/github/CenterForTheBuiltEnvironment/pythermalcomfort/requirements/?branch=master

.. |codecov| image:: https://codecov.io/github/CenterForTheBuiltEnvironment/pythermalcomfort/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/CenterForTheBuiltEnvironment/pythermalcomfort

.. |version| image:: https://img.shields.io/pypi/v/pythermalcomfort.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/pythermalcomfort

.. |wheel| image:: https://img.shields.io/pypi/wheel/pythermalcomfort.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/pythermalcomfort

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/pythermalcomfort.svg
    :alt: Supported versions
    :target: https://pypi.org/project/pythermalcomfort

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/pythermalcomfort.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/pythermalcomfort

.. |commits-since| image:: https://img.shields.io/github/commits-since/CenterForTheBuiltEnvironment/pythermalcomfort/v0.5.2.svg
    :alt: Commits since latest release
    :target: https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort/compare/v0.5.2...master



.. end-badges

Package to calculate several thermal comfort indices (e.g. PMV, PPD, SET, adaptive) and convert physical variables.

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

Examples

https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort/tree/master/examples

YouTube tutorials

https://www.youtube.com/watch?v=MLQecKYQFvg&list=PL9DCOjERDBeGGVBBlaEWMMDoS_06MZcto


Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
