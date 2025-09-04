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
from typing import List, Optional, Union

from parameterized import parameterized
from tokenicer import Tokenicer
from transformers import AutoTokenizer


class TestPadToken(unittest.TestCase):
    @parameterized.expand(
        [
            ('/monster/data/model/Llama-3.2-1B-Instruct', '<|reserved_special_token_0|>', ['<|reserved_special_token_0|>']),
            ('/monster/data/model/Phi-3-mini-4k-instruct', '<|endoftext|>'),
            ('/monster/data/model/Llama-3.2-1B-Instruct', '<|finetune_right_pad_id|>'),
            ('/monster/data/model/Qwen2.5-0.5B-Instruct', '<|fim_pad|>'),
            ('/monster/data/model/Qwen2-VL-2B-Instruct', '<|endoftext|>'),
            ('/monster/data/model/gemma-2-9b', '<pad>'),
            ('/monster/data/model/Hymba-1.5B-Instruct', '<unk>', None, True),
            ('/monster/data/model/Mistral-7B-Instruct-v0.2', '<unk>'),
            ('/monster/data/model/Yi-Coder-1.5B-Chat', '<unk>'),
            ('/monster/data/model/LongCat-Flash-Chat', '<longcat_pad>', None, True),
            (AutoTokenizer.from_pretrained('/monster/data/model/glm-4-9b-chat-hf'), '<|endoftext|>')
        ]
    )
    def test_pad_token(self,
                       tokenizer_or_path: str,
                       expect_pad_token: str,
                       pad_tokens: Optional[List[Union[str, int]]] = None,
                       trust_remote: bool = False
                       ):
        tokenicer = Tokenicer.load(tokenizer_or_path, trust_remote_code=trust_remote)

        if pad_tokens is not None:
            tokenicer.auto_fix_pad_token(pad_tokens=pad_tokens)

        self.assertEqual(
            tokenicer.tokenizer.pad_token,
            expect_pad_token,
            msg=f"Expected pad_token: `{expect_pad_token}`, actual=`{tokenicer.tokenizer.pad_token}`."
        )
