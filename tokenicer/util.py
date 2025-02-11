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

from typing import Union, List, Optional
from transformers import AutoConfig, PretrainedConfig
import string


def candidate_ids(token_list: List[Union[str, int]], vocab: dict) -> List[Optional[int]]:
    token_ids = []
    for item in token_list:
        if isinstance(item, str):
            val = vocab.get(item)
            if val is not None:
                token_ids.append(val)
        elif isinstance(item, int):
            if 0 <= item < len(vocab):
                token_ids.append(item)
    return token_ids


def candidate_id(token_list: List[Union[str, int]], vocab: dict) -> Optional[int]:
    token_ids = candidate_ids(token_list=token_list, vocab=vocab)
    return token_ids[0] if token_ids else None


def config_path(obj) -> Optional[str]:
    path = getattr(obj, "name_or_path", None)
    return path


def auto_config(path, trust_remote) -> Optional[PretrainedConfig]:
    config = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote)
    model_config = None
    if isinstance(config, PretrainedConfig):
        model_config = config
    return model_config


def all_special_characters():
    # Get punctuation characters
    punctuation_chars = string.punctuation

    # Get whitespace characters (such as space, newline, tab, etc.)
    whitespace_chars = string.whitespace

    # Common mathematical symbols and operators (manually added)
    math_and_operators = "+-*/=<>%&^|!~"

    # Combine all special characters into a single string
    all_special_chars = punctuation_chars + whitespace_chars + math_and_operators

    # Return the combined string
    return all_special_chars

