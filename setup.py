#!/usr/bin/env python
from __future__ import annotations

import re
from os.path import dirname, join

from setuptools import find_packages, setup


def read(*names, **kwargs):
    with open(
        join(dirname(__file__), *names), encoding=kwargs.get("encoding", "utf8")
    ) as fh:
        return fh.read()


setup(
    name="pythermalcomfort",
    version="3.0.1",
    license="MIT",
    description=(
        "pythermalcomfort is a comprehensive toolkit for calculating "
        "thermal comfort indices, heat/cold stress metrics, and thermophysiological responses. "
        "It supports multiple models, including PMV, PPD, adaptive comfort, SET, "
        "UTCI, Heat Index, Wind Chill Index, and Humidex. "
        "The package also includes thermophysiological models like the two-node (Gagge) and multinode (JOS-3) models "
        "to estimate physiological responses such as core temperature, skin temperature, and skin wettedness. "
    ),
    long_description="{}\n{}".format(
        re.compile("^.. start-badges.*^.. end-badges", re.M | re.S).sub(
            "", read("README.rst")
        ),
        re.sub(":[a-z]+:`~?(.*?)`", r"``\1``", read("CHANGELOG.rst")),
    ),
    long_description_content_type="text/x-rst",
    author="Federico Tartarini",
    author_email="cbecomforttool@gmail.com",
    url="https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Education",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Topic :: Utilities",
    ],
    project_urls={
        "Documentation": "https://pythermalcomfort.readthedocs.io/",
        "Changelog": "https://pythermalcomfort.readthedocs.io/en/latest/changelog.html",
        "Issue Tracker": (
            "https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort/issues"
        ),
    },
    keywords=[
        "thermal comfort",
        "pmv",
        "heat stress",
        "cold stress",
        "thermal sensation",
        "thermal physiology",
        "meteorology",
        "climate analysis",
        "discomfort",
        "comfort",
        "thermal environment",
        "built environment",
    ],
    python_requires=">=3.8.0",
    install_requires=[
        "scipy",
        "numba",
        "numpy",
        "setuptools",
    ],
    extras_require={
        "dev": ["pytest", "sphinx"],
    },
    entry_points={
        "console_scripts": [
            "pythermalcomfort = pythermalcomfort.cli:main",
        ]
    },
)
