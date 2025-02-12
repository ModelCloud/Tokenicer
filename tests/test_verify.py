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
from tokenicer import Tokenicer
from tokenicer.const import VERIFY_JSON_FILE_NAME
import tempfile


class TestVerify(unittest.TestCase):

    def test_save(self):
        model_path = "/monster/data/model/Qwen2.5-0.5B-Instruct"
        tokenicer = Tokenicer.load(model_path)

        with tempfile.TemporaryDirectory() as tmpdir:
            tokenicer.save(tmpdir)
            verify_json_path = os.path.join(tmpdir, VERIFY_JSON_FILE_NAME)
            result = os.path.isfile(verify_json_path)
            self.assertTrue(result, f"Save verify file failed: {verify_json_path} does not exist.")

            result = tokenicer.verify(verify_json_path)
            self.assertTrue(result, f"Verification failed")





