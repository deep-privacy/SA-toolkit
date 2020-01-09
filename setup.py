import setuptools
from setuptools import setup
import os

_here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(_here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

version = {}
with open(os.path.join(_here, "damped", "version.py")) as f:
    exec(f.read(), version)

setup(
    name="damped",
    version=version["__version__"],
    long_description_content_type="text/markdown",
    description=(
        "A PyTorch Domain Adaptation Module for Privacy Enable and Distributed learning"
    ),
    long_description=long_description,
    author="Pierre CHAMPION",
    author_email="prr.champion@gmail.com",
    url="https://github.com/deep-privacy/damped",
    license="TODO",
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=["torch>=1.0.1", "scikit-learn>=0.20.1", "pandas>=0.23.4"],
    #   no scripts in this example
    #   scripts=['bin/a-script'],
    include_package_data=True,
    classifiers=[
        "Operating System :: Linux",
        "Intended Audience :: Audio Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.6",
    ],
)
