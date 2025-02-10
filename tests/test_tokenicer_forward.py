from tokenicer import Tokenicer
from parameterized import parameterized
import unittest

class TestTokenicer(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.pretrained_model_id = "/monster/data/model/Qwen2.5-0.5B-Instruct/"
        self.tokenizer = Tokenicer.load(tokenizer_or_path=self.pretrained_model_id)
        self.tokenizer.auto_assign_pad_token()
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
            msg=f"Expected property result='{expect_token}' but got '{result}'."
        )

    def test_tokenicer_encode(self):
         input_ids = self.tokenizer.encode(self.example, add_special_tokens=False)
         self.assertEqual(
             input_ids,
             self.expect_input_ids,
             msg=f"Expected input_ids='{self.expect_input_ids}' but got '{input_ids}'."
         )

    def test_tokenicer_decode(self):
        example = self.tokenizer.decode(self.expect_input_ids, skip_special_tokens=True)
        self.assertEqual(
            self.example,
            example,
            msg=f"Expected example='{self.example}' but got '{example}'."
        )