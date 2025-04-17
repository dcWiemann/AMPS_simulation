from setuptools import setup, find_packages

setup(
    name="amps_simulation",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "sympy",
        "scipy"
    ],
    python_requires=">=3.8",
)