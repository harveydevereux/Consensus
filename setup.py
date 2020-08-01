import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="consensus",
    version="0.0.1",
    author="Harvey Devereux",
    author_email="harveydevereux@googlemail.com",
    description="Package for Consensus algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/harveydevereux/Consensus",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
