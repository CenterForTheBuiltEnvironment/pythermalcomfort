# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.extlinks",
    "sphinx.ext.ifconfig",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_rtd_theme",
]
source_suffix = ".rst"
master_doc = "index"
project = "pythermalcomfort"
year = "2019"
author = "Federico Tartarini"
copyright = "{0}, {1}".format(year, author)
version = release = "2.10.0"

autodoc_typehints = "none"
pygments_style = "trac"
templates_path = ["."]
extlinks = {
    "issue": (
        "https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort/issues/%s",
        "issue %s",
    ),
    "pr": (
        "https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort/pull/%s",
        "PR %s",
    ),
}
# on_rtd is whether we are on readthedocs.org
on_rtd = os.environ.get("READTHEDOCS", None) == "True"

html_theme = "sphinx_rtd_theme"

html_use_smartypants = True
html_last_updated_fmt = "%b %d, %Y"
html_split_index = False
html_sidebars = {
    "**": ["searchbox.html", "globaltoc.html", "sourcelink.html"],
}
html_short_title = "%s-%s" % (project, version)

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False
