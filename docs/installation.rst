============
Installation
============

To install the `pythermalcomfort` package, follow the instructions below.
Options are provided for typical users and for developers who want to work on
the project locally.

Using pip (recommended)
=======================

Install the latest release from PyPI:

.. code-block:: bash

    pip install pythermalcomfort

This will install the package and its runtime dependencies.

From source (stable or development)
==================================

Clone the repository and install from the local copy. This is useful if you
want to inspect the code or install a development branch:

.. code-block:: bash

    git clone https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort.git
    cd pythermalcomfort
    pip install .

Editable/developer install
==========================

For active development, create a virtual environment, install the package in
editable mode and install development dependencies. Editable install lets you
modify the source code and use the updated package without reinstalling.

.. code-block:: bash

    python -m venv .venv
    source .venv/bin/activate        # use .venv\Scripts\activate on Windows
    pip install -e .[dev]

The project uses tox for running the test and docs environments. After the
editable install you can still use tox to validate across targeted Python
versions.

Running tests locally
=====================

Run the full test matrix with tox (may take some time):

.. code-block:: bash

    tox

Run a single environment (replace ``py312`` with the desired env):

.. code-block:: bash

    tox -e py312

Or run pytest directly for faster iteration during development:

.. code-block:: bash

    pytest -q

Dependencies and optional extras
================================

Runtime dependencies (installed automatically via pip):

- numpy
- scipy
- pandas

Development extras (used by maintainers and contributors):

- ruff (linting/formatting)
- docformatter (docstring formatting)
- sphinx and Sphinx extensions (docs)
- pytest (testing)

If you installed via ``pip install -e .[dev]`` these will be installed for you.

Verifying the installation
==========================

A quick smoke test to confirm the package imports and basic functions work:

.. code-block:: python

    >>> from pythermalcomfort.models import pmv_ppd_iso
    >>> result = pmv_ppd_iso(tdb=25, tr=25, vr=0.1, rh=50, met=1.4, clo=0.5)
    >>> print(result.pmv, result.ppd)

You can also run a short pytest command to ensure tests pass locally:

.. code-block:: bash

    pytest tests/test_pmv_ppd.py -q

Troubleshooting
===============

- If installation fails with binary wheel or compilation errors, ensure you
  have an up-to-date pip, setuptools, and wheel:

.. code-block:: bash

    pip install --upgrade pip setuptools wheel

- On macOS, if you get errors building native extensions, ensure Xcode
  command line tools are installed: ``xcode-select --install``.

- If tox environments fail due to missing interpreters, install the desired
  Python versions locally or use pyenv to manage them.

- If documentation build fails, check the Sphinx requirements in
  ``docs/requirements.txt`` and install them into your environment.

Further resources
=================

* Full documentation and examples: https://pythermalcomfort.readthedocs.io
* Contribution guidelines: see ``CONTRIBUTING.rst`` in the project root
