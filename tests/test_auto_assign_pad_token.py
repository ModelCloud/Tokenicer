import unittest
from parameterized import parameterized
from typing import Union, List

from tokenicer import Tokenicer


class TestAutoAssignPadToken(unittest.TestCase):
    NATIVE_TOKENIZER_PATH = "/monster/data/model/Llama-3.2-1B-Instruct"

    @parameterized.expand(
        [
            ([], '<|finetune_right_pad_id|>'),
            (['<|reserved_special_token_0|>'], '<|reserved_special_token_0|>')
        ]
    )
    def test_auto_assign_pad_token(self, pad_tokens: List[Union[str, int]], assign_pad_token_result: int):
        tokenicer = Tokenicer.load(tokenizer_or_path=self.NATIVE_TOKENIZER_PATH)
        tokenicer.auto_assign_pad_token(pad_tokens=pad_tokens)
        self.assertEqual(
            tokenicer.tokenizer.pad_token,
            assign_pad_token_result,
            msg=f"Expected pad_token='{assign_pad_token_result}' but got '{tokenicer.tokenizer.pad_token}' for pad_tokens={pad_tokens}."
        )