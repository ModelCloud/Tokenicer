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
