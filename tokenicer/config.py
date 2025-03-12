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

from typing import List, Optional, Union, Any, Dict
from dataclasses import dataclass
from enum import Enum


class ValidateDataFormat(Enum):
    SIMPLE = "simple"


@dataclass
class ValidateData:
    format: ValidateDataFormat = ValidateDataFormat.SIMPLE
    input: Union[str, Any] = None
    output: List[int] = None

    def __post_init__(self):
        if self.input is None:
            self.input = []

        if self.output is None:
            self.output = []


@dataclass
class ValidateMeta:
    validator: str = None
    uri: str = None

    def __post_init__(self):
        if self.validator is None:
            from .version import __version__

            self.validator = f"tokenicer:{__version__}"

        if self.uri is None:
            self.uri = "https://github.com/ModelCloud/Tokenicer"


@dataclass
class ValidateConfig:
    meta: Optional[ValidateMeta] = None
    data: List[ValidateData] = None

    def __post_init__(self):
        if self.meta is None:
            self.meta = ValidateMeta()

        if self.data is None:
            self.data = []

    def to_dict(self):
        dataset_dict = [
            {
                "format": data.format.value,
                "input": data.input,
                "output": data.output,
            }
            for data in self.data
        ]

        meta_dict = {"validator": self.meta.validator, "uri": self.meta.uri}

        return {"meta": meta_dict, "data": dataset_dict}

    @classmethod
    def from_dict(cls, data: Dict):
        meta_data = data.get("meta", {})
        data_list = data.get("data", [])
        meta = ValidateMeta(**meta_data) if meta_data else None
        validate_data = [ValidateData(**item) for item in data_list]
        return cls(meta=meta, data=validate_data)
