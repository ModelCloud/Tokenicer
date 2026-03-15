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

import json
import os
from typing import List, Optional, Union

from transformers import AutoConfig, PretrainedConfig


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
    try:
        config = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote)
    except Exception:
        # Tokenizer-only bundles and malformed local configs should not block tokenizer loading.
        return None
    model_config = None
    if isinstance(config, PretrainedConfig):
        model_config = config
    return model_config


def has_custom_tokenizer_code(path) -> bool:
    if not isinstance(path, str) or not os.path.isdir(path):
        return False

    tokenizer_config_path = os.path.join(path, "tokenizer_config.json")
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path, "r", encoding="utf-8") as f:
            tokenizer_config = json.load(f)
        auto_map = tokenizer_config.get("auto_map")
        if auto_map is not None:
            # AutoTokenizer auto_map entries mean the checkpoint expects custom tokenizer code.
            if isinstance(auto_map, dict) and auto_map.get("AutoTokenizer") is not None:
                return True
            if isinstance(auto_map, (list, tuple)) and auto_map:
                return True

    return any(name.startswith("tokenization") and name.endswith(".py") for name in os.listdir(path))


def tokenizer_special_token_overrides(path) -> dict:
    if not isinstance(path, str) or not os.path.isdir(path):
        return {}

    tokenizer_config_path = os.path.join(path, "tokenizer_config.json")
    if not os.path.exists(tokenizer_config_path):
        return {}

    with open(tokenizer_config_path, "r", encoding="utf-8") as f:
        tokenizer_config = json.load(f)

    tokenizer_class = tokenizer_config.get("tokenizer_class")
    if tokenizer_class not in {"LlamaTokenizer", "LlamaTokenizerFast"}:
        return {}

    overrides = {}
    # Some local Llama-family checkpoints serialize empty strings for required special tokens.
    defaults = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
    }
    for key, default_value in defaults.items():
        if tokenizer_config.get(key, None) == "":
            overrides[key] = default_value

    return overrides


def tokenizer_config_dict(path) -> dict:
    if not isinstance(path, str) or not os.path.isdir(path):
        return {}

    tokenizer_config_path = os.path.join(path, "tokenizer_config.json")
    if not os.path.exists(tokenizer_config_path):
        return {}

    with open(tokenizer_config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenizer_class_name(path) -> Optional[str]:
    tokenizer_config = tokenizer_config_dict(path)
    tokenizer_class = tokenizer_config.get("tokenizer_class")
    if isinstance(tokenizer_class, str) and tokenizer_class:
        return tokenizer_class
    return None


def custom_tokenizer_class_ref(path) -> Optional[str]:
    tokenizer_config = tokenizer_config_dict(path)
    auto_map = tokenizer_config.get("auto_map")
    if isinstance(auto_map, dict):
        auto_tokenizer_ref = auto_map.get("AutoTokenizer")
        if isinstance(auto_tokenizer_ref, str):
            return auto_tokenizer_ref
        if isinstance(auto_tokenizer_ref, (list, tuple)):
            for ref in auto_tokenizer_ref:
                if isinstance(ref, str) and ref:
                    return ref
    return None
