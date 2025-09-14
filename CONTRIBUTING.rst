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
* Detailed steps to reproduce the bug.

Documentation Improvements
==========================

pythermalcomfort can always use more documentation, whether as part of the official docs, in docstrings, or even on the web in blog posts, articles, and such.

Issues, Features and Feedback
=============================

The best way to send feedback is to submit an `issue <https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort/issues>`_.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible to make it easier to implement.
* Remember that this is a volunteer-driven project, and code contributions are welcome :)

Development
===========

To set up `pythermalcomfort` for local development:

1. Fork `pythermalcomfort <https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort>`_ (look for the "Fork" button).
2. Clone your fork locally. Fetch and pull all updates from the master branch before you do anything:

.. code-block:: bash

    git clone git@github.com:CenterForTheBuiltEnvironment/pythermalcomfort.git

3. Create a branch for local development. The naming rules for new branches are as follows:

* For a new feature: `Feature/feature_name_here`
* For a bug fix: `Fix/bug_name_here`
* For documentation: `Documentation/doc_name_here`

You can create a branch locally using the following command. Make sure you only push updates to this new branch:

.. code-block:: bash

    git checkout -b name-of-your-bugfix-or-feature

Now you can make your changes locally.

4. When you're done making changes, run all the checks and docs builder with tox in one command:

.. code-block:: bash

    tox
    tox -e docs
    tox -e py312

5. Format the code and lint it:

.. code-block:: bash

    autopep8 --in-place --max-line-length 88 --select E501 --aggressive pythermalcomfort/*.py
    ruff check --fix
    ruff format
    docformatter -r -i --wrap-summaries 88 --wrap-descriptions 88 pythermalcomfort

5. Commit your changes and push your branch to GitHub:

.. code-block:: bash

    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature

6. Submit a pull request after you have done all your modifications and tested your work. The pull request should include a detailed description of your work:

* What this pull request is about.
* How you tested your work.
* Whether this work affects other components in the project.

Pull Request Guidelines
-----------------------

If you need a code review or feedback while developing, just make the pull request.

For merging, you should:

1. Include passing tests (run ``tox``).
2. Update documentation when there's new API, functionality, etc.
3. Add a note to ``CHANGELOG.rst`` about the changes.
4. Add yourself to ``AUTHORS.rst``.

Tips
----

To run a subset of tests:

.. code-block:: bash

    tox -e envname -- pytest -k test_myfeature

To run all the test environments in *parallel*:

.. code-block:: bash

    tox --parallel

To Add a Function
-----------------

Use this comprehensive checklist when adding a new function to pythermalcomfort:

**Function Development:**

1. **Create function in new file**: Add your function to a new Python file in `pythermalcomfort/models/` with a meaningful, descriptive name that follows the existing naming conventions.

2. **Ensure meaningful naming**: Use clear, descriptive names for the function itself. This should reflect and be the same as the filename.

3. **Variable naming consistency**: Follow the established variable naming patterns used throughout the project (e.g., `tdb` for dry-bulb temperature, `tr` for mean radiant temperature, `rh` for relative humidity). You can find a list of common variable names in BaseInputs in `classes_input.py`.

4. **Add comprehensive docstring**: Include a complete docstring with:

    - Clear function description explaining what the model calculates
    - Add relevant standards or research papers
    - **Parameters** section with type hints, units, and detailed descriptions
    - **Returns** section describing the output dataclass
    - **Examples** section with practical usage examples showing both single values and arrays

5. **Add input validation**: Include input validation like other functions:

    - `limit_inputs: bool = True` parameter for standard applicability limits
    - `round_output: bool = True` parameter for output rounding
    - Use appropriate input validation classes (see `classes_input.py`)

6. **Add type annotations**: Provide complete type annotations for all parameters and return values, supporting both single values and arrays (e.g., `float | list[float]`).

7. **Ensure array support**: Make sure your function accepts and properly handles numpy arrays, Pandas Series, and lists as inputs, following the pattern in existing functions.

8. **Must return a dataclass**: Create and return a dataclass that inherits from `AutoStrMixin` (see `classes_return.py` for examples).

**Validation and Quality:**

9. **Check applicability limits**: Implement proper checking for the standard applicability limits of your model and return `nan` values when inputs are outside valid ranges. You can strictly ensure that applicability limits are enforced in the input validation classes. For example, this is necessary to ensure that the air speed is higher than 0 m/s Some other limits can be a bit more flexible, like temperature ranges. See the `pmv_ppd_iso.py` function for reference, where the limits are not strictly enforced but `nan` is returned when inputs are outside the valid range. The `limit_inputs` parameter can be used to toggle this behavior.

10. **Handle edge cases**: Test and handle edge cases appropriately, ensuring robust behavior with various input combinations. For example, ensure that when relative humidity is 0%, the function behaves correctly and does not produce errors. Or that inputs are not negative when they shouldn't be or are not string values when they should be numeric. All these checks should be done in the input validation classes. See `scale_wind_speed_log.py` for an example of handling edge cases.

**Documentation and Integration:**

11. **Add to documentation**: Add your function to the documentation in `docs/documentation/models.rst` using the `autofunction` directive. Or add it to `utilities_functions.rst` if it's a utility function.

12. **Update version**: Bump the minor version number in the appropriate configuration files. This should not be done manually but using `bump-my-version bump minor` command. Major version bumps should be reserved for breaking changes only.

13. **Add to changelog**: Add an entry to `CHANGELOG.rst` describing the new function and the changes made.

**Testing:**

14. **Test the function**: Write comprehensive tests in `tests/test_XXXX.py` that cover:

    - Basic functionality with reference values
    - Array input handling
    - Edge cases and error conditions
    - Input validation behavior
    - Output format and dataclass structure
    - XXXX is the name of your function file without the .py extension

**Before Submission:**

15. Run the full test suite: `tox -e pyXX` where `XX` is your Python version (e.g., `312`)
16. Format and lint your code:

.. code-block:: bash

    autopep8 --in-place --max-line-length 88 --select E501 --aggressive pythermalcomfort/*.py
    ruff check --fix
    ruff format
    docformatter -r -i --wrap-summaries 88 --wrap-descriptions 88 pythermalcomfort

17. Build documentation to ensure it renders correctly: `tox -e docs`

**Reference Template:**

Use existing functions like `pmv_ppd_iso.py` as a template to ensure your function follows all established patterns for input validation, array handling, output formatting, and documentation style.
