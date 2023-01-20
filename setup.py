from setuptools import setup

# Workaround to get pip to install dependencies. pyproject.toml field not working.
setup(
    install_requires = [
        "setuptools>=42",
        "pandas>=1.2.0",
        "intervaltree==3.1.0",
        "psutil>=5.9.0"
    ]
)
