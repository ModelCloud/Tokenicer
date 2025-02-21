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
import os
import unittest
from typing import Type, Dict

from parameterized import parameterized
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerBase, Qwen2TokenizerFast

from tokenicer import Tokenicer


class Test(unittest.TestCase):

    @parameterized.expand(
        [
            ("/monster/data/model/Qwen2.5-0.5B-Instruct/", Qwen2TokenizerFast),
        ]
    )
    def test(self, model_path:str, expected_type:Type[PreTrainedTokenizerBase]):
        tokenicer = Tokenicer.load(model_path)

        self.assertIsInstance(tokenicer, expected_type)
        self.assertIsInstance(AutoTokenizer.from_pretrained(model_path), expected_type)
