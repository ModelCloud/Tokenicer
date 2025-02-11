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

from collections import namedtuple

DEFAULT_PAD_TOKENS = [
        "<|finetune_right_pad_id|>",
        "<|pad|>",
        "<pad>",
        "<|unk|>",
        "<unk>"
]

TOKEN_TUPLE = namedtuple("TokenTuple", ["token", "token_id"])

MODEL_PAD_TOKEN_MAP = {
        "llama": TOKEN_TUPLE(token='<|finetune_right_pad_id|>', token_id=128004),
        "qwen2_5_vl": TOKEN_TUPLE(token='<|vision_pad|>', token_id=151654),
        "qwen2_vl": TOKEN_TUPLE(token='<|vision_pad|>', token_id=151654),
        "qwen2": TOKEN_TUPLE(token='<|fim_pad|>', token_id=151662),
        "deepseek_v3": TOKEN_TUPLE(token='<｜▁pad▁｜>', token_id=2),
        "mpt": TOKEN_TUPLE(token='<|padding|>', token_id=1)
}

VERIFY_JSON_FILE_NAME = "tokenizer_verify.jsonl"
VERIFY_ENCODE_PARAMS = {"return_tensors": "pt", "add_special_tokens": False}

INPUT_KEY = "input"
TENSOR_KEY = "tensor"

VERIFY_DATASETS = [
        "Sure! I'd be happy to help. What kind of writing prompt are you looking for?",
        "Certainly! A comma (,) is used to separate items in a list, e.g., 'I bought apples, bananas, and oranges.' A semicolon (;) links related independent clauses, e.g., 'I have a meeting tomorrow; I need to prepare.' A colon (:) introduces a list or explanation, e.g., 'Here are the items you need: pen, paper, and ink.'",
        "Let's break it down:\n\n1. 3.14159265359 + 2.71828182846 = 5.85987448205\n2. 5.6 * 2.3 = 12.88\n3. The square root of 123.456 is approximately 11.1111047355\n\nWould you like to explore more complex calculations? I can also work with exponents (e.g., 2^10 or 5.1^3.2).",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
]