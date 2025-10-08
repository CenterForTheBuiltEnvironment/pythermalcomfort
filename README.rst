.. image:: https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort/raw/development/docs/images/pythermalcomfort-3-short.png
  :align: center
  :alt: pythermalcomfort logo

================
pythermalcomfort
================

The ``pythermalcomfort`` Python package is a comprehensive toolkit for calculating
thermal comfort indices, heat/cold stress metrics, and thermophysiological
responses based on international standards and peer-reviewed research. Designed
for researchers, engineers, and building-science professionals, it simplifies
complex calculations while promoting accuracy and standards compliance.

Cite pythermalcomfort
=====================

If you use ``pythermalcomfort`` in your research, please cite it as follows:

.. code-block:: text

   Tartarini, F., Schiavon, S., 2020.
   pythermalcomfort: A Python package for thermal comfort research.
   SoftwareX 12, 100578.
   https://doi.org/10.1016/j.softx.2020.100578

Key Features
============

- **Thermal Comfort Calculations**:
  Supports multiple models, including **PMV**, **PPD**, **adaptive comfort**, and
  **SET**.
- **Heat and Cold Stress Indices**:
  Calculate **UTCI**, **Heat Index**, **Wind Chill Index**, and **Humidex**.
- **Thermophysiological Models**:
  Includes the **two-node (Gagge)** and **multinode (JOS-3)** models to estimate
  physiological responses such as core temperature, skin temperature, and skin
  wettedness.
- **Standards Compliance**:
  Implements calculations based on **ASHRAE 55**, **ISO 7730**, **EN 16798**, and
  more.
- **Ease of Use**:
  Intuitive API for seamless integration into Python projects.
- **Extensive Documentation**:
  Detailed guides, examples, and tutorials.
- **Active Development**:
  Regularly updated with new features, improvements, and bug fixes.
- **Open Source**:
  Licensed under the MIT License.

Why Use pythermalcomfort?
=========================

- **Accurate Assessments**:
  Reliable thermal comfort and stress evaluations for diverse environments.
- **Time-Saving**:
  Automates complex calculations, saving time and effort.
- **Versatility**:
  Useful for building science, HVAC design, environmental design, thermal
  physiology, sports science, and biometeorology.
- **Enhanced Decision-Making**:
  Supports data-driven decisions for HVAC systems, building performance, and
  occupant comfort.

Installation
============

Install ``pythermalcomfort`` via pip:

.. code-block:: bash

   pip install pythermalcomfort

For advanced installation options, refer to the
`Installation Instructions <https://pythermalcomfort.readthedocs.io/en/latest/installation.html>`_.

Requirements
============

A typical environment includes:

- Python 3.9+ (or the minimum version used in CI)
- NumPy, SciPy, pandas (installed automatically as package dependencies)
- Optional: Matplotlib / other plotting libraries for examples and visualizations

Quick Start
===========

Get started with ``pythermalcomfort`` in just a few lines of code:

.. code-block:: python

   from pythermalcomfort.models import pmv_ppd_iso, utci

   # Calculate PMV and PPD using ISO 7730 standard
   result = pmv_ppd_iso(
       tdb=25,  # Dry Bulb Temperature in °C
       tr=25,  # Mean Radiant Temperature in °C
       vr=0.1,  # Relative air speed in m/s
       rh=50,  # Relative Humidity in %
       met=1.4,  # Metabolic rate in met
       clo=0.5,  # Clothing insulation in clo
       model="7730-2005"  # Year of the ISO standard
   )
   print(f"PMV: {result.pmv}, PPD: {result.ppd}")

   # Calculate UTCI for heat stress assessment
   utci_value = utci(tdb=30, tr=30, v=0.5, rh=50)
   print(utci_value)

For more examples and detailed usage, check out models and indices in the models section of the documentation.

Support pythermalcomfort
========================

If you find this project useful, consider supporting the maintainers. Maintaining
an open-source scientific package requires time for development, review, CI,
and user support.

Ways to support
---------------

- Sponsor via GitHub: https://github.com/sponsors/FedericoTartarini
- Contribute code, tests, or documentation: open a PR against `pythermalcomfort`
- Report bugs or request features with a minimal reproduction in `issues`
- Help with testing, translations, or reviewing pull requests
- Star or share the project to increase visibility

Any support—financial or contribution-based—is appreciated and helps keep the
project healthy.

Contribute
==========

We welcome contributions! Whether reporting a bug, suggesting a feature, or
submitting a pull request, your input helps make ``pythermalcomfort`` better for
everyone. See the `contributing guide <https://pythermalcomfort.readthedocs.io/en/latest/contributing.html>`_ for full contribution instructions.

Quick checklist
---------------

* Open an issue first for larger features to discuss scope and design.
* Fork the repo and create a feature branch for your work.
* Add tests for new behavior and run the test suite locally.
* Run linters and formatters and fix reported issues.
* Update documentation and changelog entries for public API changes.
* Submit a clear, focused pull request referencing any related issues.

Common commands
---------------

.. code-block:: bash

    # clone your fork and add upstream remote
    git clone git@github.com:your-username/pythermalcomfort.git
    cd pythermalcomfort
    git remote add upstream git@github.com:CenterForTheBuiltEnvironment/pythermalcomfort.git
    git fetch upstream

    # create a branch and work on it
    git checkout -b Feature/awesome-feature

    # run the full test matrix (may be slow)
    tox

    # run a single test env locally (replace py312 with the env you want)
    tox -e py312

    # run a subset of pytest tests
    pytest -k test_name_fragment

    # fix linting/formatting
    ruff check --fix
    ruff format
    docformatter --in-place --wrap-summaries 88 --wrap-descriptions 88 pythermalcomfort/*.py

    # commit and push
    git add .
    git commit -m "feat: short description of change"
    git push origin Feature/awesome-feature

Where to get help
-----------------

* Open an issue on GitHub with a minimal reproduction for bugs.
* Ask questions in PR comments for implementation guidance.
* See the `contribution guidelines <https://pythermalcomfort.readthedocs.io/en/latest/contributing.html>`_ for detailed guidance on testing,
  documentation, and changelog expectations.
* For API reference and examples, consult the online docs:
  https://pythermalcomfort.readthedocs.io/en/latest/

Documentation
-------------

Detailed docs, examples and API references are available at:
https://pythermalcomfort.readthedocs.io/en/latest/

License
=======

``pythermalcomfort`` is released under the MIT License.

=====
Stats
=====

.. start-badges

.. list-table::
    :stub-columns: 1

    * - Documentation
      - |docs|
    * - License
      - |license|
    * - Downloads
      - |downloads|
    * - Tests
      - | |codecov|
        | |tests|
    * - Package
      - | |version| |wheel|
        | |supported-ver|
        | |package-health|

.. |tests| image:: https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort/actions/workflows/build-test-publish.yml/badge.svg
    :target: https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort/actions/workflows/build-test-publish.yml
    :alt: Tests to ensure pythermalcomfort works on different Python versions and OS

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

.. |codecov| image:: https://codecov.io/github/CenterForTheBuiltEnvironment/pythermalcomfort/coverage.svg?branch=master
    :alt: Coverage Status
    :target: https://codecov.io/github/CenterForTheBuiltEnvironment/pythermalcomfort

.. |version| image:: https://img.shields.io/pypi/v/pythermalcomfort.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/pythermalcomfort

.. |wheel| image:: https://img.shields.io/pypi/wheel/pythermalcomfort.svg
    :alt: pythermalcomfort wheel
    :target: https://pypi.org/project/pythermalcomfort

.. |supported-ver| image:: https://img.shields.io/pypi/pyversions/pythermalcomfort.svg
    :alt: Supported versions
    :target: https://pypi.org/project/pythermalcomfort

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/pythermalcomfort.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/pythermalcomfort

.. end-badges
