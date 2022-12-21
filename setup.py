import sys
import os.path as op

from setuptools import setup, find_packages


# get the version (don't import mne here, so dependencies are not needed)
version = None
with open(op.join("pyriemann", "_version.py"), "r") as fid:
    for line in (line.strip() for line in fid):
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip("'")
            break
if version is None:
    raise RuntimeError("Could not determine version")

# "open" does not provide the parameter "encoding" for python<3
is_python3 = sys.version_info.major >= 3
kwargs_open = {"encoding": "utf8"} if is_python3 else {}
with open('README.md', 'r', **kwargs_open) as fid:
    long_description = fid.read()

setup(
    name="pyriemann",
    version=version,
    description="Riemannian Geometry for python",
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
    python_requires=">=3.7",
    install_requires=["numpy", "scipy", "scikit-learn", "joblib", "pandas"],
    extras_require={
        "docs": [
            "sphinx-gallery",
            "sphinx-bootstrap_theme",
            "numpydoc",
            "mne",
            "seaborn",
        ],
        "tests": ["pytest", "seaborn", "flake8"],
    },
    zip_safe=False,
)
