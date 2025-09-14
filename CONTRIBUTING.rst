============
Contributing
============

Contributions are welcome and greatly appreciated!
Every bit helps, and credit will always be given.

Bug Reports
===========

When `reporting a bug <https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort/issues>`_, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug (minimal reproducible example preferred).
* Relevant error messages and a shortened traceback.

Documentation Improvements
==========================

pythermalcomfort can always use more documentation: docs, docstrings, examples,
and tutorials are all valuable.

Issues, Features and Feedback
=============================

The best way to send feedback is to open an `issue <https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort/issues>`_.

If you are proposing a feature:

* Explain in detail how it would work and why it's useful.
* Keep the scope narrow so it is easier to review and implement.
* Consider opening a discussion issue first for larger changes.

Development
===========

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

Use this checklist when adding a new function (summary — see code examples in
the repository for concrete patterns):

1. Create the function in a new file under ``pythermalcomfort/models/`` with a
   descriptive filename (match the function name).
2. Use consistent parameter names (e.g., ``tdb``, ``tr``, ``vr``, ``rh``, ``met``,
   ``clo``) and include full type annotations.
3. Add a comprehensive NumPy-style docstring with parameters, units, returns,
   examples (scalars and arrays), and references.
4. Validate inputs and implement applicability limits (use input dataclasses).
5. Support numpy arrays, lists, and pandas.Series for inputs.
6. Return a dataclass (see ``classes_return.py``) for structured outputs.
7. Add tests covering correct values, arrays, broadcasting, and edge cases.
8. Add the function to the documentation (``docs/...rsts``) via ``autofunction``.
9. Add a changelog entry and update versioning as appropriate.

Validation and Quality
----------------------

* Respect applicability limits and return ``nan`` for out-of-range cases when
  ``limit_inputs`` is used.
* Add robust handling for edge cases (e.g., zero/negative values where
  inappropriate).
* Follow existing patterns (see ``pmv_ppd_iso.py`` and utilities).

Reference Template
------------------

Use existing functions (for example ``pmv_ppd_iso.py``) as templates for
structure, validation, documentation, and tests.

Contribute
==========

We welcome contributions of all kinds: bug reports, documentation, tests,
translations, and code. See the repository CONTRIBUTING and docs for more
details.

Where to get help
-----------------

* Open an issue on GitHub with a minimal reproduction for bugs.
* Ask questions in PR comments for implementation guidance.
* See the CONTRIBUTING.rst file for development and testing guidelines.
* For API reference and examples, consult the online docs:
  https://pythermalcomfort.readthedocs.io/en/latest/

Tips
----

* Open an issue first for larger features to discuss scope and design.
* Keep PRs focused and small where possible.
* Include tests and documentation for public API changes.

License
=======

pythermalcomfort is released under the MIT License.
