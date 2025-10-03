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
    "myst_nb",  # render/execute notebooks
    "nbsphinx",  # render/execute notebooks inline
    # "myst_parser",     # optional if you use MyST markdown elsewhere
    # "sphinx_gallery.gen_gallery",  # convert example .py -> gallery with plots
]
source_suffix = ".rst"
master_doc = "index"
project = "pythermalcomfort"
year = "2025"
author = "Federico Tartarini"
project_copyright = f"{year}, {author}"
version = release = "3.6.0"

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

html_theme = "pydata_sphinx_theme"

napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_use_param = False

html_title = f"pythermalcomfort {version}"
html_short_title = f"{project}-{version}"

html_use_smartypants = True
html_last_updated_fmt = "%b %d, %Y"
html_split_index = False

html_sidebars = {
    "**": ["sidebar-nav-bs"],
    "installation": [],
    "contributing": [],
    "authors": [],
}

html_theme_options = {
    "icon_links": [
        {
            # Label for this link
            "name": "GitHub",
            # URL where the link will redirect
            "url": "https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort",  # required
            # Icon class (if "type": "fontawesome"), or path to local image (if "type": "local")
            "icon": "fa-brands fa-square-github",
            # The type of image to be used (see below for details)
            "type": "fontawesome",
        },
        {
            "name": "LinkedIn",
            "url": "https://www.linkedin.com/in/federico-tartarini",  # required
            "icon": "fa-brands fa-linkedin",
            "type": "fontawesome",
        },
        {
            "name": "Google Scholar",
            "url": "https://scholar.google.com/citations?view_op=list_works&hl=en&hl=en&user=QcamSPwAAAAJ",
            "icon": "fa-brands fa-google-scholar",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/pythermalcomfort/",
            "icon": "fa-brands fa-python",
        },
    ],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "content_footer_items": ["last-updated"],
}

# # sphinx-gallery config (example .py -> gallery)
# sphinx_gallery_conf = {
#     "examples_dirs": "../examples",   # where your example .py scripts live
#     "gallery_dirs": "auto_examples",  # output dir inside docs build
# }

# MyST-NB config (executing notebooks and embedding outputs)
nb_execution_mode = (
    "auto"  # "force" to always execute, "auto" to execute if outputs missing
)
nb_execution_timeout = 180
myst_enable_extensions = ["dollarmath", "amsmath"]
