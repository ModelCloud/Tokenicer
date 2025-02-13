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

import unittest
from tokenicer import Tokenicer


class TestModelConfig(unittest.TestCase):
    def test_model_config(self):
        model_path = "/monster/data/model/mpt-7b-instruct"
        tokenicer = Tokenicer.load(model_path)

        expect_bos_token_id = 0
        expect_eos_token_id = 0

        self.assertEqual(
            tokenicer.model_config.bos_token_id,
            expect_bos_token_id,
            msg=f"Expected bos_token_id: `{expect_bos_token_id}`, actual=`{tokenicer.model_config.bos_token_id}`.",
        )

        self.assertEqual(
            tokenicer.model_config.eos_token_id,
            expect_eos_token_id,
            msg=f"Expected eos_token_id: `{expect_eos_token_id}`, actual=`{tokenicer.model_config.eos_token_id}`.",
        )
