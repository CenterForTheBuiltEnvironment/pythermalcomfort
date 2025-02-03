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

    black .
    autopep8 --in-place --max-line-length 88 --select E501 --aggressive pythermalcomfort/*.py
    ruff check --fix
    ruff format
    docformatter --in-place --wrap-summaries 88 --wrap-descriptions 88 pythermalcomfort/*.py

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
^^^^^^^^^^^^^^^^^

1. Add the function to the Python file `pythermalcomfort/models/` and document it.
2. Add any related functions that are used by your function either in `pythermalcomfort/utilities.py`. See existing code as examples.
3. Ensure that all new functions accept arrays as input and return a dataclass. You can use the code in `pmv_ppd_iso.py` as a template.
4. Test your function by writing a test in `tests/test_XXXX.py`. Test it by running `tox -e pyXX` where `XX` is the Python version you want to use, e.g., `37`.
5. Add `autofunction` to `doc.reference.pythermalcomfort.py`.
