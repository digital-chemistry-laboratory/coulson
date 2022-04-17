import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="coulson",
    version="0.0.1",
    author="Kjell Jorner",
    author_email="kjell.jorner@gmail.com",
    description="Program for Hückel molecular orbital theory and aromaticity",
    long_description=long_description,
    long_description_content_type="text/x-markdown",
    url="",
    packages=setuptools.find_packages(),
    install_requires=["numpy", "networkx", "scipy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
