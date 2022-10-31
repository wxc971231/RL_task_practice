from setuptools import setup, find_packages

setup(
    name="MyGymExamples",
    version="0.0.1",
    install_requires=["gym==0.26.2", "pygame==2.1.0"],
    packages=find_packages()
)