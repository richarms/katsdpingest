#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="katsdpingest",
    description="Karoo Array Telescope Data Capture",
    author="Simon Ratcliffe",
    packages=find_packages(),
    package_data={'': ['ingest_kernels/*.mako']},
    include_package_data=True,
    scripts=[
        "scripts/ingest.py",
        "scripts/ingest_autotune.py"
    ],
    setup_requires=['katversion'],
    install_requires=[
        'manhole',
        'numpy',
        'scipy',
        'scikits.fitting',
        'spead2>=0.3.0',
        'katcp',
        'katpoint',
        'katsdpsigproc',
        'katsdpdisp',
        'katsdptelstate'
    ],
    zip_safe=False,
    use_katversion=True
)
