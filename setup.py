from setuptools import setup, find_packages

setup(
    name="cache-edit",
    version="0.1.0",
    packages=find_packages(exclude=["tests", "tests.*", "legacy", "legacy.*"]),
    python_requires=">=3.8",
)
