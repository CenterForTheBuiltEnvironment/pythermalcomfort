#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import io
import re
from glob import glob
from os.path import basename
from os.path import dirname
from os.path import join
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def read(*names, **kwargs):
    with io.open(
        join(dirname(__file__), *names),
        encoding=kwargs.get('encoding', 'utf8')
    ) as fh:
        return fh.read()


setup(
    name='pythermalcomfort',
    version='1.3.6',
    license='MIT',
    description='Package to calculate several thermal comfort indices (e.g. PMV, PPD, SET, adaptive) and convert physical variables. Please cite us if you use this package: Tartarini, F., Schiavon, S., 2020. pythermalcomfort: A Python package for thermal comfort research. SoftwareX 12, 100578. https://doi.org/10.1016/j.softx.2020.100578',
    long_description='%s\n%s' % (
        re.compile('^.. start-badges.*^.. end-badges', re.M | re.S).sub('', read('README.rst')),
        re.sub(':[a-z]+:`~?(.*?)`', r'``\1``', read('CHANGELOG.rst'))
    ),
    author='Federico Tartarini',
    author_email='cbecomforttool@gmail.com',
    url='https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Unix',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Scientific/Engineering',
        'Topic :: Utilities',
    ],
    project_urls={
        'Documentation': 'https://pythermalcomfort.readthedocs.io/',
        'Changelog': 'https://pythermalcomfort.readthedocs.io/en/latest/changelog.html',
        'Issue Tracker': 'https://github.com/CenterForTheBuiltEnvironment/pythermalcomfort/issues',
    },
    keywords=[
        'thermal comfort', 'pmv', 'ppd', 'building design', 'compliance', 'thermal environment', 'buit environment'
    ],
    python_requires='>=3.6.*',
    install_requires=[
        'scipy',
        'numba'
        # eg: 'aspectlib==1.1.1', 'six>=1.7',
    ],
    extras_require={
        # eg:
        #   'rst': ['docutils>=0.11'],
        #   ':python_version=="2.6"': ['argparse'],
    },
    entry_points={
        'console_scripts': [
            'pythermalcomfort = pythermalcomfort.cli:main',
        ]
    },
)
