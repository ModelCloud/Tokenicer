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

from tokenicer import Tokenicer
from parameterized import parameterized
import unittest

class TestTokenicer(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.pretrained_model_id = "/monster/data/model/Qwen2.5-0.5B-Instruct/"
        self.tokenizer = Tokenicer.load(self.pretrained_model_id)
        self.example = 'Test Case String'
        self.expect_input_ids = [2271, 11538, 923]

    def test_tokenicer_func(self):
        input_ids = self.tokenizer(self.example)['input_ids']
        self.assertEqual(
            input_ids,
            self.expect_input_ids,
            msg=f"Expected input_ids='{self.expect_input_ids}' but got '{input_ids}'."
        )

    @parameterized.expand(
        [
            ('eos_token', "<|im_end|>"),
            ('pad_token', "<|fim_pad|>"),
            ('vocab_size', 151643)
        ]
    )
    def test_tokenicer_property(self, property, expect_token):
        if property == 'eos_token':
            result = self.tokenizer.eos_token
        elif property == 'pad_token':
            result = self.tokenizer.pad_token
        elif property == 'vocab_size':
            result = self.tokenizer.vocab_size

        self.assertEqual(
            result,
            expect_token,
            msg=f"Expected {property}: '{expect_token}', actual='{result}'."
        )

    def test_tokenicer_encode(self):
         input_ids = self.tokenizer.encode(self.example, add_special_tokens=False)
         self.assertEqual(
             input_ids,
             self.expect_input_ids,
             msg=f"Expected input_ids: '{self.expect_input_ids}', actual='{input_ids}'."
         )

    def test_tokenicer_decode(self):
        example = self.tokenizer.decode(self.expect_input_ids, skip_special_tokens=True)
        self.assertEqual(
            self.example,
            example,
            msg=f"Expected example: '{self.example}', actual='{example}'."
        )