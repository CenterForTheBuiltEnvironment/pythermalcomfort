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
===================================

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

* Full documentation and examples: `Full documentation <https://pythermalcomfort.readthedocs.io/en/latest>`_
* Contribution guidelines: see `Contributing Instructions <https://pythermalcomfort.readthedocs.io/en/latest/contributing.html>`_ in the project root

.. _using-from-r:

Using pythermalcomfort from R
=============================

You can call the Python package from R using the reticulate package.
Two common workflows are shown below: using a virtualenv or a conda environment.
Adjust commands to your OS and Python installation.

Virtualenv (recommended when using virtualenv)
----------------------------------------------

.. code-block:: r

    # install reticulate if needed
    install.packages("reticulate")

    library(reticulate)

    # create and install pythermalcomfort into a virtualenv
    virtualenv_create("r-pythermal")
    virtualenv_install("r-pythermal", packages = c("pythermalcomfort"))

    # activate the virtualenv for this R session
    use_virtualenv("r-pythermal", required = TRUE)

    # import the package and call functions
    ptc <- import("pythermalcomfort")
    pmv_ppd_iso <- ptc$models$pmv_ppd_iso
    res <- pmv_ppd_iso(tdb = 25, tr = 25, vr = 0.1, rh = 50,
                      met = 1.4, clo = 0.5)
    # access results (attributes of the Python return object)
    res$pmv
    res$ppd

Conda environment
-----------------

.. code-block:: r

    library(reticulate)

    # create a conda env and install pythermalcomfort
    conda_create("r-pythermal")
    conda_install("r-pythermal", packages = c("pythermalcomfort"), channel = "defaults")

    # use the conda env in this session
    use_condaenv("r-pythermal", required = TRUE)

    # import and use as above
    ptc <- import("pythermalcomfort")
    pmv_ppd_iso <- ptc$models$pmv_ppd_iso

Vectorized inputs and conversions
---------------------------------

.. code-block:: r

    library(reticulate)

    # for vector inputs, convert R vectors to Python objects when needed
    tdb <- c(20, 25, 30)
    tr <- rep(25, 3)
    vr <- rep(0.1, 3)
    rh <- rep(50, 3)

    # r_to_py is optional; reticulate will attempt automatic conversion
    res <- pmv_ppd_iso(tdb = r_to_py(tdb), tr = r_to_py(tr),
                      vr = r_to_py(vr), rh = r_to_py(rh),
                      met = 1.2, clo = 0.5)

    # extract scalar/vector results
    pmv_vec <- py_to_r(res$pmv)
    ppd_vec <- py_to_r(res$ppd)

Notes and tips
--------------

- If you prefer installing from R directly into the active Python environment, reticulate offers py_install():
  reticulate::py_install("pythermalcomfort", envname = "r-pythermal", pip = TRUE)
- Access Python objects' attributes with the $ operator (e.g., res$pmv).
- Use py_to_r() to convert numpy arrays or Python lists into R vectors or lists.
- If reticulate cannot find the correct Python, set RETICULATE_PYTHON to a specific interpreter path before loading reticulate:
  Sys.setenv(RETICULATE_PYTHON = "/path/to/python")
- See the reticulate guide for details: https://rstudio.github.io/reticulate/

Further resources
=================

* Full documentation and examples: `Full documentation <https://pythermalcomfort.readthedocs.io/en/latest>`_
* reticulate documentation: https://rstudio.github.io/reticulate/
