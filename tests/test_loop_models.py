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

from parameterized import parameterized

from tokenicer import Tokenicer


MODEL_DIR = "/monster/data/model"

class Test(unittest.TestCase):
    model_list = [
        os.path.join(MODEL_DIR, f)
        for f in os.listdir(MODEL_DIR)
        if os.path.isdir(os.path.join(MODEL_DIR, f))
        and any(f == "tokenizer_config.json" for f in os.listdir(os.path.join(MODEL_DIR, f)))
        ]

    @parameterized.expand(model_list)
    def test(self, model_path):
        tokenicer = Tokenicer.load(model_path, trust_remote_code=True)
