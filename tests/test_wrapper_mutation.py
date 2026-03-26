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

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from tokenicer import Tokenicer


class TestWrapperMutation(unittest.TestCase):
    def test_setting_padding_side_updates_wrapped_tokenizer(self):
        backend = Tokenizer(WordLevel({"<pad>": 0, "hello": 1}, unk_token="<pad>"))
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            pad_token="<pad>",
            unk_token="<pad>",
        )
        tokenicer = Tokenicer.load(tokenizer)

        tokenicer.padding_side = "left"
        tokenicer.truncation_side = "left"

        self.assertEqual(tokenicer.padding_side, "left")
        self.assertEqual(tokenicer.tokenizer.padding_side, "left")
        self.assertEqual(tokenicer.truncation_side, "left")
        self.assertEqual(tokenicer.tokenizer.truncation_side, "left")
