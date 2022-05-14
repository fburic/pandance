import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

project_urls = {
    "Documentation": "https://pandance.readthedocs.io",
    "Source Code": "https://github.com/fburic/pandance",
}

setuptools.setup(
    name='pandance',
    version='0.1.0',
    packages=['pandance'],
    description='',
    long_description=long_description,
    project_urls = project_urls,
    long_description_content_type="text/markdown",
    license='BSD',
    author='Filip Buric',
    author_email='',
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Database",
        "Topic :: Scientific/Engineering ",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only"
    ],
    python_requires='>=3.8',
)