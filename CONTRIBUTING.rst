============
Contributing
============

Contributions are welcome and greatly appreciated!
Every bit helps, and credit will always be given.

Bug, Feature, Discussions, and Documentation Issues
===================================================

Bug Reports
-----------

When `reporting a bug <https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort/issues>`_, please complete
the issue template with as much relevant information as possible, including:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug (minimal reproducible example preferred).
* Relevant error messages and a shortened traceback.

Documentation Improvements
--------------------------

pythermalcomfort can always use more documentation: docs, docstrings, examples,
and tutorials are all valuable. Hence, if you find something missing or unclear,
please consider contributing an improvement!
The documentation is built using Sphinx and reStructuredText, all the files are
located in the `docs/` folder.

Features, Feedback, and Discussions
-----------------------------------

The best way to suggest a new feature is to open an `issue <https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort/issues>`_.
Alternatively, you can start a discussion in the `Discussions section <https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort/discussions>`_.

If you are proposing a feature, please use the `Feature request` template and:

* Explain in detail how it would work and why it's useful.
* Keep the scope narrow so it is easier to review and implement.
* Consider opening a discussion first for larger changes.

Contributing - Code
===================

This section explains how to set up your development environment,
run tests, format code, and contribute code changes via pull requests (PRs).
If you want to contribute a new model or function, please see also the detailed guide
at the end of this file.

Setting up your development environment:

1. Fork the repository on GitHub and clone your fork locally:

.. code-block:: bash

    git clone git@github.com:your-username/pythermalcomfort.git
    cd pythermalcomfort
    git remote add upstream git@github.com:CenterForTheBuiltEnvironment/pythermalcomfort.git
    git fetch upstream

2. Set up a virtual environment and install dependencies, you should have Python 3.12+ installed and `pipenv <https://pipenv.pypa.io/en/latest/>`_ available:

.. code-block:: bash

    pip install pipenv
    pipenv sync --dev

3. Create a feature branch (use the naming rules below):

.. code-block:: bash

    git checkout -b Feature/short-description
    # or for bug fixes
    git checkout -b Fix/short-description

Branch naming
-------------

* Feature branches: ``Feature/your-feature-name``
* Bugfix branches: ``Fix/your-bug-name``
* Documentation: ``Documentation/doc-name``

Pre-commit & local checks
-------------------------

Run these checks before committing and opening a PR:

.. code-block:: bash

    # run tests quickly (subset)
    pytest -k test_fragment

To run the rest with tox and a specific Python version:

.. code-block:: bash

    tox -e pyXXX  # replace XXX with your Python version, e.g., 312

    # alternatively run full test suite via tox, you will need to have different Python versions installed
    tox

Formatting and linting

.. code-block:: bash

    # formatting & linting (preferred)
    ruff check --fix
    ruff format
    docformatter -r -i --wrap-summaries 88 --wrap-descriptions 88 pythermalcomfort
    # optional; install pre-commit hooks (pre-commit install)
    # pre-commit run --all-files

Pull request checklist
----------------------

When opening a pull request, include:

* A clear summary of the change and motivation.
* Tests for new behavior and updates for any affected tests.
* Relevant documentation updates (docstrings or docs/).
* A CHANGELOG entry (if applicable).
* Add yourself to AUTHORS.rst (optional).

Contributing a New Function
===========================

Above we have already covered how to set up your development environment,
run tests, and format/lint your code.

Use this checklist and follow the detailed steps to add a new, well-tested,
documented function consistent with the project's conventions.
Please have a look at existing functions for reference, for example in
``pythermalcomfort/models/pmv_ppd_iso.py`` and associated tests and utilities.

Quick checklist (use before opening a PR)

- [ ] Function added under an appropriate module, if it is a thermal index/model, please add under ``pythermalcomfort/models/``.
- [ ] Input dataclass created/updated with validation in __post_init__. See ``pythermalcomfort/classes_input.py``.
- [ ] Return dataclass created/updated if applicable. See ``pythermalcomfort/classes_return.py``.
- [ ] NumPy-style docstring with units, examples, and applicability limits.
- [ ] Tests added (scalars, arrays, broadcasting, invalid inputs).
- [ ] Documentation (autofunction/autodoc) updated.
- [ ] CHANGELOG and AUTHORS updated (if applicable).
- [ ] All tests pass and formatting/linting applied.
- [ ] Add the function to ``__init__.py`` which is located in the ``pythermalcomfort/models/`` folder.
- [ ] The function should accept both scalar and vectorized inputs (lists, numpy arrays) and return outputs of matching shape.

Step-by-step guide
------------------

1) Pick the module/function location
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- If function is a new model, add under: ``pythermalcomfort/models/<module_name>.py``
- If generic utility, consider: ``pythermalcomfort/utilities.py``
- Use a descriptive module name.

2) Implement the function
^^^^^^^^^^^^^^^^^^^^^^^^^

When implementing the function, follow these guidelines:

- Keep it simple, documented, and readable.
- Accept both scalar and vectorized inputs (lists, numpy arrays).
- Use numpy vectorized operations for performance (e.g., ``np.log``) rather than ``math``.
- Add a NumPy-style docstring including: Parameters, Raises, Returns, Examples, References.
- Check in the BaseInputs how inputs are typically named, typed, and validated.
- Example skeleton for a new function:

.. code-block:: python

     # pythermalcomfort/models/my_func.py
     import numpy as np
     from dataclasses import dataclass
     from pythermalcomfort.classes_input import MyFuncInputs
     from pythermalcomfort.classes_return import DataClassResult

     def my_func(x: float | np.ndarray) -> DataClassResult:
        """Short description.

        Add more detailed description here in a new paragraph and include
        citations.

        Parameters
        ----------
        x: float | np.ndarray
            Description of x (include units).

        Returns
        -------
        DataClassResult
            Description of return value (include units).

        Raises
        ------
        ValueError
            If x is negative.

        Examples
        --------
        .. code-block:: python

            from pythermalcomfort.models import my_func

            tdb = 25
            result = my_func(tdb)
            print(result)  # Expected output: ...
        """

        # validate and normalize inputs via dataclass
        MyFuncInputs(x=x)

        x_arr = np.atleast_1d(x)
        if np.any(x_arr < 0):
            raise ValueError("x must be non-negative")

        # simple example computation
        value = x_arr * 2
        return DataClassResult(result=value)


3) Create / update an input dataclass for functions of models.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As previously mentioned, input validation is centralized via dataclasses, see ``pythermalcomfort/classes_input.py``. Follow these steps:

- Add input dataclasses to ``pythermalcomfort/classes_input.py``
- Put type checks and physical/applicability checks in ``__post_init__``. See for example, ``THIInputs``.
- Types are automatically validated by the dataclass, so focus on value checks (e.g., ranges, non-negativity).

4) Return types and classes_return
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As a convention, functions should return structured dataclasses where applicable.

- Return a dataclass from ``classes_return.py`` to provide structured outputs.
- Keep the public API clear and documented.

5) Tests
^^^^^^^^

- Add tests under ``tests/test_<function>.py``.
- Example pytest skeleton:

.. code-block:: python

   import numpy as np
   import pytest
   from pythermalcomfort.models.my_func import my_func

   def test_my_func_scalar():
       assert my_func(1.0) == pytest.approx(2.0)

   def test_my_func_array():
       arr = np.array([1.0, 2.0])
       out = my_func(arr)
       assert out.shape == arr.shape

   def test_my_func_invalid():
       with pytest.raises(ValueError):
           my_func(-1.0)


- Keep tests deterministic and small. Use numpy.testing where helpful.

6) Documentation & autodoc
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Add a short example to the function docstring.
- Add an ``.. autofunction:: pythermalcomfort.models.my_func.my_func`` entry in the relevant docs source file (e.g., ``docs/reference.rst`` or the file that collects API references).
- Add the return data class to the documentation if new.

7) CHANGELOG and AUTHORS
^^^^^^^^^^^^^^^^^^^^^^^^

- Add a short line to the changelog describing the new function.
- Optionally add yourself to AUTHORS.rst when contributing a new feature.

8) Formatting, linting and tests locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Apply project formatters and linters:

.. code-block:: bash

    ruff check --fix
    ruff format
    docformatter -r -i --wrap-summaries 88 --wrap-descriptions 88 pythermalcomfort

    # optional; install pre-commit hooks (pre-commit install)

- Run tests:

.. code-block:: bash

    tox -e pyXXX  # replace XXX with your Python version, e.g., 312

9) Open a PR
^^^^^^^^^^^^

When opening a pull request, follow these guidelines:

- Title: short descriptive title (e.g., "feat: add my_func for X calculation")
- Include in PR description:
   - What the function does and why.
   - Applicability limits and physical constraints.
   - How it was tested (mention key tests).
   - Notes about numeric stability or edge cases.
- Ensure CI passes and add reviewers as appropriate.

PR checklist (add to your PR description)

- [ ] New tests added and passing.
- [ ] Docstring updated with examples and applicability limits.
- [ ] Documentation (autofunction) updated if public API changed.
- [ ] CHANGELOG updated (if applicable).
- [ ] Linting and formatting applied.
- [ ] All CI checks pass.

Where to get help
-----------------

* Open an issue on GitHub with a minimal reproduction for bugs.
* Ask questions in PR comments for implementation guidance.
* See the CONTRIBUTING.rst file for development and testing guidelines.
* For API reference and examples, consult the online docs:
  `Full documentation <https://pythermalcomfort.readthedocs.io/en/latest>`_

License
=======

pythermalcomfort is released under the MIT License.
