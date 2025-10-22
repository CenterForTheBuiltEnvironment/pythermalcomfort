============
Contributing
============

Pythermalcomfort provides models for thermal comfort analysis.
Contributions are welcome and greatly appreciated!

Bug Reports
===========

When `reporting a bug <https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort/issues>`_, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug (minimal reproducible example preferred).
* Relevant error messages and a shortened traceback.
* Pythermalcomfort version (`python -m pip show pythermalcomfort`)
Documentation Improvements
==========================

pythermalcomfort can always use more documentation: docs, docstrings, examples,
and tutorials are all valuable.
Follow NumPy-style docstrings for consistency.
Docs are located in `docs/`.

Issues, Features and Feedback
=============================

The best way to send feedback is to open an `issue <https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort/issues>`_.

If you are proposing a feature:

* Explain in detail how it would work and why it's useful.
* Keep the scope narrow so it is easier to review and implement.
* Consider opening a discussion issue first for larger changes.

Contributing Code
=================

To set up `pythermalcomfort` for local development:

1. Fork the repository on GitHub and clone your fork locally:

.. code-block:: bash

    git clone git@github.com:your-username/pythermalcomfort.git
    cd pythermalcomfort
    git remote add upstream git@github.com:CenterForTheBuiltEnvironment/pythermalcomfort.git
    git fetch upstream

2. Create a feature branch (use the naming rules below):

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

    # run full test suite via tox
    tox

    # formatting & linting (preferred)
    ruff check --fix
    ruff format
    docformatter -r -i --wrap-summaries 88 --wrap-descriptions 88 pythermalcomfort
    # optional: black or other formatters if preferred by maintainers
    # optional; install pre-commit hooks (pre-commit install)

Pull request checklist
----------------------

When opening a pull request, include:

* A clear summary of the change and motivation.
* Tests for new behavior and updates for any affected tests.
* Relevant documentation updates (docstrings or docs/).
* A CHANGELOG entry (if applicable).
* Add yourself to AUTHORS.rst (optional).

Running tests
-------------

To run tests locally:

.. code-block:: bash

    # run all tests
    pytest

    # run a subset by keyword
    pytest -k test_name_fragment

    # run the CI matrix locally (may be slow)
    tox

To run a single tox environment:

.. code-block:: bash

    tox -e py312

Formatting and linting
----------------------

Recommended commands before pushing:

.. code-block:: bash

    ruff check --fix
    ruff format
    docformatter -r -i --wrap-summaries 88 --wrap-descriptions 88 pythermalcomfort/*.py

(If your editor or CI uses other tools like black or isort, follow the project's configured pre-commit hooks.)

Committing and pushing
----------------------

.. code-block:: bash

    git add .
    git commit -m "feat: short description of change"
    git push origin Feature/short-description

Submit a pull request on GitHub from your branch to the main repository.

To Add a Function
-----------------

Use this checklist and follow the detailed steps to add a new, well-tested,
documented function consistent with the project's conventions.

Quick checklist (use before opening a PR)

- [ ] Implementation added under an appropriate module.
- [ ] Input dataclass created/updated with validation in __post_init__.
- [ ] NumPy-style docstring with units, examples and applicability limits.
- [ ] Tests added (scalars, arrays, broadcasting, invalid inputs).
- [ ] Documentation (autofunction/autodoc) updated.
- [ ] CHANGELOG and AUTHORS updated (if applicable).
- [ ] All tests pass and formatting/linting applied.

Step-by-step guide
^^^^^^^^^^^^^^^^^^

1) Pick the module location
   - If function is a domain model, add under: ``pythermalcomfort/models/<module_name>.py``
   - If generic utility, consider: ``pythermalcomfort/utilities.py``

2) Implement the function
   - Keep it small, pure and documented.
   - Use numpy for numeric ops (e.g., ``np.log``) rather than ``math``.
   - Add a NumPy-style docstring including: Parameters, Raises, Returns, Examples, References.
   - Example skeleton:

   .. code-block:: python

     # pythermalcomfort/models/my_func.py
     import numpy as np

     def my_func(x: float | np.ndarray) -> float | np.ndarray:
         """Short description.

         Parameters
         ----------
             x: value in meters.

         Returns
         -------
             Dataclass with fields

         Raises
         ------
             ValueError: if x is negative.

         Examples
         --------
             >>> my_func(1.0)
             2.0
         """

         x_arr = np.asarray(x)
         if np.any(x_arr < 0):
             raise ValueError("x must be non-negative")
         return x_arr * 2

3) Create / update an input dataclass (if applicable)

   - Add input dataclasses to ``pythermalcomfort/classes_input.py`` or next to the function module to centralize validation.
   - Put type checks and physical/applicability checks in ``__post_init__``.
   - Convert pandas Series to lists/arrays before validation.
   - Use ``validate_type(name, allowed_types)`` for type validation.
   - Example pattern:

   .. code-block:: python

     @dataclass
     class MyFuncInputs(BaseInputs):
         x: float | int | list | np.ndarray = None

         def __post_init__(self):
             super().__post_init__()
             # validate types (raises TypeError)
             validate_type(self.x, "x", (float, int, list, np.ndarray))
             # normalize to numpy array for vectorized ops
             self.x = np.asarray(self.x)
             # physical checks (raises ValueError)
             if np.any(self.x < 0):
                 raise ValueError("x must be non-negative")
             # broadcasting checks if multiple array fields exist


4) Return types and classes_return

   - When consistent with other functions, return a dataclass from ``classes_return.py`` to provide structured outputs.
   - Keep the public API clear and documented.

5) Tests

   - Add tests under ``tests/test_<function>.py``.
   - Cover:

     - Scalar inputs (single values).
     - Vectorized inputs (lists, numpy arrays).
     - Broadcasting behavior and consistent output shapes.
     - Invalid inputs (TypeError and ValueError cases).
     - Edge cases (zeros, very small/large inputs that affect numeric stability).

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

   - Add a short example to the function docstring.
   - Add an ``.. autofunction:: pythermalcomfort.models.my_func.my_func`` entry in the relevant docs source file (e.g., ``docs/reference.rst`` or the file that collects API references).
   - If a larger example/tutorial is needed, add an rst under ``docs/`` and include usage examples (scalar and vectorized).

7) CHANGELOG and AUTHORS

   - Add a short line to the changelog describing the new function.
   - Optionally add yourself to AUTHORS.rst when contributing a new feature.

8) Formatting, linting and tests locally

   - Apply project formatters and linters:

   .. code-block:: bash

      ruff check --fix
      ruff format
      docformatter -r -i --wrap-summaries 88 --wrap-descriptions 88 pythermalcomfort

   - Run tests:

   .. code-block:: bash

      pytest -q

9) Open a PR

- Title: short descriptive title (e.g., "feat: add my_func for X calculation")
- Include in PR description:
   - What the function does and why.
   - Applicability limits and physical constraints.
   - How it was tested (mention key tests).
   - Notes about numeric stability or edge cases.
- Ensure CI passes and add reviewers as appropriate.

Recommended validation rules (common to many functions)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Non-negativity: ensure physical quantities that must be >= 0 are validated.
- Domain checks: avoid taking logs/roots of non-positive numbers.
- Shape/broadcasting: if arrays are accepted, verify shapes are compatible.
- Units: document the expected units and validate/convert where needed.
- Error types: use TypeError for wrong types, ValueError for invalid values.

PR checklist (add to your PR description)

- [ ] New tests added and passing.
- [ ] Docstring updated with examples and applicability limits.
- [ ] Documentation (autofunction) updated if public API changed.
- [ ] CHANGELOG updated (if applicable).
- [ ] Linting and formatting applied.
- [ ] All CI checks pass.

Examples and reference patterns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Use existing functions (e.g., ``pmv_ppd_iso.py`` and utilities) as reference
  for structure, validation, and tests.
- For mathematical operations prefer numpy and vectorized routines.
- For input classes follow the BaseInputs metadata-driven pattern and centralize
  shared conversions and type checks there.

Where to get help
-----------------

* Open an issue on GitHub with a minimal reproduction for bugs.
* Ask questions in PR comments for implementation guidance.
* See the CONTRIBUTING.rst file for development and testing guidelines.
* For API reference and examples, consult the online docs:
  `Full documentation <https://pythermalcomfort.readthedocs.io/en/latest>`_

Tips
----

* Open an issue first for larger features to discuss scope and design.
* Keep PRs focused and small where possible.
* Include tests and documentation for public API changes.

License
=======

pythermalcomfort is released under the MIT License.
See 'add link'
