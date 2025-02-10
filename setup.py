from setuptools import setup, find_packages
from pathlib import Path

__version__ = "0.0.1-dev"

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="tokenicer",
    version=__version__,
    author="ModelCloud",
    author_email="qubitium@modelcloud.ai",
    description="A nicer tokenizer",
    long_description=(Path(__file__).parent / "README.md").read_text(encoding="UTF-8"),
    long_description_content_type="text/markdown",
    url="https://github.com/ModelCloud/Tokenicer",
    packages=find_packages(),
    install_requires=requirements,
    platform=["linux", "windows", "darwin"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3",
)