# Copyright 2025 ModelCloud.ai
# Copyright 2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages
from pathlib import Path

__version__ = "0.0.2"

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="tokenicer",
    version=__version__,
    author="ModelCloud",
    author_email="qubitium@modelcloud.ai",
    description="A (nicer) tokenizer you want to use for model `inference` and `training`: with all known peventable `gotchas` normalized or auto-fixed.",
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
