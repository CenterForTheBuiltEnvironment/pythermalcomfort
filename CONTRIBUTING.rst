============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

Bug reports
===========

When `reporting a bug <https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort/issues>`_ please include:

    * Your operating system name and version.
    * Any details about your local setup that might be helpful in troubleshooting.
    * Detailed steps to reproduce the bug.

Documentation improvements
==========================

pythermalcomfort could always use more documentation, whether as part of the
official pythermalcomfort docs, in docstrings, or even on the web in blog posts,
articles, and such.

Feature requests and feedback
=============================

The best way to send feedback is to file an issue at https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that code contributions are welcome :)

Development
===========

To set up `pythermalcomfort` for local development:

1. Fork `pythermalcomfort <https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort>`_
   (look for the "Fork" button).
2. Clone your fork locally. Fetch and pull all the updates from the master branch before you do anything:

.. code-block::

    git clone git@github.com:CenterForTheBuiltEnvironment/pythermalcomfort.git

3. Create a branch for local development. The naming rule for new branch are, as follows:

    * If this update is for a new feature Feature/feature_name_here
    * If this update is for bug fix Fix/bug_name_here
    * If this update is for documentation Documentation/doc_name_here

You can create a branch locally using the following command. Make sure you only push updates to this new branch only:

.. code-block::

    git checkout -b name-of-your-bugfix-or-feature

Now you can make your changes locally.

4. When you're done making changes run all the checks and docs builder with tox one command:

.. code-block::

    tox

5. Commit your changes and push your branch to GitHub:

.. code-block::

    git add .
    git commit -m "Your detailed description of your changes."
    git push origin name-of-your-bugfix-or-feature

6. Submit a pull request after you have done all your modifications and tested your work. The pull request should include a detailed description of your work:

    * What this pull request is about
    * Have you tested your work
    * Will this work affect other component in the product

Pull Request Guidelines
-----------------------

If you need some code review or feedback while you're developing the code just make the pull request.

For merging, you should:

1. Include passing tests (run ``tox``).
2. Update documentation when there's new API, functionality etc.
3. Add a note to ``CHANGELOG.rst`` about the changes.
4. Add yourself to ``AUTHORS.rst``.

Tips
----

To run a subset of tests:

.. code-block::

    tox -e envname -- pytest -k test_myfeature

To run all the test environments in *parallel* (you need to ``pip install detox``):

.. code-block::

    detox

To add a function
^^^^^^^^^^^^^^^^^

1. Add the function to the python file `pythermalcomfort/models/` and document it.
2. Add any related functions that are used by your function either in `pythermalcomfort/utilities.py` or `src/pythermalcomfort/psychrometrics.py`. See existing code as example.
3. Test your function by writing a test in `tests/test_XXXX.py`. Test it by running tox -e pyXX where XX is the Python version you want to use, e.g. 37
4. Add `autofunction` to `doc.reference.pythermalcomfort.py`.
