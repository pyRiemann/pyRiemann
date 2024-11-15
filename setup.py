import os.path as op
import sys

from setuptools import find_packages, setup

version = None
with open(op.join("pyriemann", "_version.py"), "r") as fid:
    for line in (line.strip() for line in fid):
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip("'")
            break
if version is None:
    raise RuntimeError("Could not determine version")

is_python3 = sys.version_info.major >= 3
kwargs_open = {"encoding": "utf8"} if is_python3 else {}
with open('README.md', 'r', **kwargs_open) as fid:
    long_description = fid.read()

description = "Machine learning for multivariate data with Riemannian geometry"

setup(
    name="pyriemann",
    version=version,
    description=description,
    url="https://pyriemann.readthedocs.io",
    author="Alexandre Barachant",
    author_email="alexandre.barachant@gmail.com",
    license="BSD (3-clause)",
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Documentation": "https://pyriemann.readthedocs.io",
        "Source": "https://github.com/pyRiemann/pyRiemann",
        "Tracker": "https://github.com/pyRiemann/pyRiemann/issues/",
    },
    platforms="any",
    python_requires=">=3.9",
    install_requires=[
        "numpy>=2.0.0",
        "scipy",
        "scikit-learn>=0.24",
        "joblib",
        "matplotlib",
    ],
    extras_require={
        "docs": [
            "sphinx",
            "sphinx-gallery",
            "sphinx-bootstrap_theme",
            "numpydoc",
            "mne",
            "seaborn",
            "pandas",
        ],
        "tests": [
            "pytest",
            "seaborn",
            "flake8"
        ],
    },
    zip_safe=False,
)
