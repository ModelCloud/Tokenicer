# Copyright 2026 ModelCloud.ai
# Copyright 2026 qubitium@modelcloud.ai
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

class TestGemma3(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.pretrained_model_id = "/monster/data/model/chatglm3-6b"  # "THUDM/chatglm3-6b"
        self.tokenizer = Tokenicer.load(self.pretrained_model_id, trust_remote_code=True)
        self.example = 'Test Case String'
        self.expect_first_token_id = 64790  # [gMASK]

    def test_encode(self):
        call_input_ids = self.tokenizer(self.example)['input_ids']

        self.assertEqual(
            call_input_ids[0],
            self.expect_first_token_id,
            msg=f"Expected func `tokenizer()` first_token_id=`{self.expect_first_token_id}`, actual=`{call_input_ids[0]}`."
        )

        encode_input_ids = self.tokenizer.encode(self.example)

        self.assertEqual(
            encode_input_ids[0],
            self.expect_first_token_id,
            msg=f"Expected func `tokenizer.encode()` first_token_id=`{self.expect_first_token_id}`, actual=`{encode_input_ids[0]}`."
        )
