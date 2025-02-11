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
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
        "",
]

VERIFY_CHAT_TEMPLAT_DATASETS = [
        [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Can you help me with a writing prompt?"},
                {"role": "assistant", "content": "Sure! I'd be happy to help. What kind of writing prompt are you looking for?"}
        ],
        [
                {"role": "system", "content": "Please provide a comprehensive response, covering various aspects—syntax, grammar, punctuation, etc. (We can discuss in detail.)"},
                {"role": "user", "content": "What is the difference between commas, semicolons, and colons? Can you give me some examples?"},
                {"role": "assistant", "content": "Certainly! A comma (,) is used to separate items in a list, e.g., 'I bought apples, bananas, and oranges.' A semicolon (;) links related independent clauses, e.g., 'I have a meeting tomorrow; I need to prepare.' A colon (:) introduces a list or explanation, e.g., 'Here are the items you need: pen, paper, and ink.'"}
        ],
        [{"role": "system", "content": ""}, {"role": "user", "content": ""}, {"role": "assistant", "content": ""}],
        [{"role": "system", "content": ""}, {"role": "user", "content": ""}, {"role": "assistant", "content": ""}],
        [{"role": "system", "content": ""}, {"role": "user", "content": ""}, {"role": "assistant", "content": ""}],
        [{"role": "system", "content": ""}, {"role": "user", "content": ""}, {"role": "assistant", "content": ""}],
        [{"role": "system", "content": ""}, {"role": "user", "content": ""}, {"role": "assistant", "content": ""}],
        [{"role": "system", "content": ""}, {"role": "user", "content": ""}, {"role": "assistant", "content": ""}],
        [{"role": "system", "content": ""}, {"role": "user", "content": ""}, {"role": "assistant", "content": ""}],
        [{"role": "system", "content": ""}, {"role": "user", "content": ""}, {"role": "assistant", "content": ""}],
]
