from setuptools import setup, find_packages

setup(
    name="vdsim",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "matplotlib",
        "pandas",
        "pytest",
    ],
    author="Fabien Lionti",
    description="Modular vehicle dynamics simulation package",
)
