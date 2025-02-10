import unittest
from parameterized import parameterized
from typing import Union, List, Optional
from transformers import AutoTokenizer

from tokenicer import Tokenicer


class TestPadToken(unittest.TestCase):
    @parameterized.expand(
        [
            ('/monster/data/model/Llama-3.2-1B-Instruct', '<|reserved_special_token_0|>', ['<|reserved_special_token_0|>']),
            ('/monster/data/model/Phi-3-mini-4k-instruct', '<unk>'),
            ('/monster/data/model/Llama-3.2-1B-Instruct', '<|finetune_right_pad_id|>'),
            ('/monster/data/model/Qwen2.5-0.5B-Instruct', '<|fim_pad|>'),
            ('/monster/data/model/Qwen2-VL-2B-Instruct', '<|vision_pad|>'),
            ('/monster/data/model/gemma-2-9b', '<pad>'),
            ('/monster/data/model/Hymba-1.5B-Instruct', '<unk>', None, True),
            ('/monster/data/model/Mistral-7B-Instruct-v0.2', '<unk>'),
            ('/monster/data/model/Yi-Coder-1.5B-Chat', '<unk>'),
            (AutoTokenizer.from_pretrained('/monster/data/model/glm-4-9b-chat-hf'), '<|endoftext|>')
        ]
    )
    def test_auto_assign_pad_token_2(self,
                                     tokenizer_or_path: str,
                                     expect_pad_token: str,
                                     pad_tokens: Optional[List[Union[str, int]]] = None,
                                     trust_remote: bool = False
                                     ):
        tokenicer = Tokenicer.load(tokenizer_or_path=tokenizer_or_path, trust_remote=trust_remote)
        tokenicer.auto_assign_pad_token(pad_tokens=pad_tokens)

        self.assertEqual(
            tokenicer.tokenizer.pad_token,
            expect_pad_token,
            msg=f"Expected pad_token='{expect_pad_token}' but got '{tokenicer.tokenizer.pad_token}' for pad_tokens={pad_tokens}."
        )