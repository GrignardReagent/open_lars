"""Shim for pip versions whose PEP 660 editable-install support is broken
(e.g. Ubuntu 22.04's stock pip 22.0.2): with this file present they fall
back to the legacy `setup.py develop` path for `pip install -e .`.
All metadata lives in pyproject.toml; modern pip ignores this file.
"""

from setuptools import setup

setup()
