import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


def parse_requirements_file(filename):
    with open(filename) as f:
        requires = [line.strip() for line in f.readlines() if line]
    return requires


setuptools.setup(
    name="dmqc",
    version="0.0.1",
    author="Mustafa Al-Rubaye",
    author_email="mustafa.al-rubaye@student.oulu.fi",
    description="A segmentation package",
    install_requires=[],
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(include=["dmqc"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
