from setuptools import setup, find_packages

setup(
    name='tidepeaks',
    version='0.1.0',
    packages=find_packages(),
    description='tidepeaks consists of a group of functions for computing a range of useful outputs from a tide gauge record. Based on the MATLAB Tide Peaks Toolbox',
    author='Ben Mildren, Karen Palmer',
    author_email='benjamin.mildren@utas.edu.au',
    url='https://github.com/ben-utas/tide-peaks-toolbox-python',
    install_requires=[
        "numpy", "scipy", "utide"
    ],
)
