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
import types
from typing import List, Optional, Union

from tokenizers import AddedToken
from transformers import AutoConfig, PretrainedConfig
from transformers.utils.hub import cached_file


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
        config_dict = _load_config_dict(path)
        if config_dict is None:
            # Tokenizer-only bundles and malformed local configs should not block tokenizer loading.
            return None

        # Transformers can be strict about custom fields, while this repo keeps
        # heterogeneous checkpoints locally. Keep the raw JSON payload for best-effort
        # config compatibility.
        config = _namespace_from_dict(config_dict)
        if "_name_or_path" not in config_dict:
            config._name_or_path = path

        # Preserve multimodal nested text config to keep compatibility with callers
        # that expect `get_text_config()`.
        text_config = getattr(config, "text_config", None)
        if text_config is not None:

            def _get_text_config():
                return text_config

            config.get_text_config = _get_text_config

        return config

    model_config = None
    if isinstance(config, PretrainedConfig):
        model_config = config
    return model_config


def _load_config_dict(path) -> Optional[dict]:
    if not isinstance(path, str) or not os.path.isdir(path):
        return None

    config_path = os.path.join(path, "config.json")
    if not os.path.exists(config_path):
        return None

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    if not isinstance(data, dict):
        return None

    return data


def _namespace_from_dict(data: dict):
    if not isinstance(data, dict):
        return data

    if isinstance(data, types.SimpleNamespace):
        return data

    values = {}
    for key, value in data.items():
        values[key] = _namespace_from_dict(value) if isinstance(value, dict) else value

    return types.SimpleNamespace(**values)


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


def _tokenizer_file(path, filename, **kwargs):
    if not isinstance(path, (str, os.PathLike)):
        return None
    path = os.fspath(path)
    if os.path.isabs(path) and not os.path.isdir(path):
        return None
    selection = {key: kwargs[key] for key in (
        "revision", "subfolder", "cache_dir", "token", "local_files_only", "force_download", "proxies", "repo_type",
    ) if key in kwargs}
    return cached_file(path, filename, _raise_exceptions_for_missing_entries=False, **selection)


def _tokenizer_metadata(path, filename, **kwargs) -> dict:
    filename = _tokenizer_file(path, filename, **kwargs)
    if filename is None:
        return {}
    with open(filename, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if not isinstance(metadata, dict):
        raise ValueError(f"Tokenicer: tokenizer metadata must be an object: {filename}")
    return metadata


def _special_token(value):
    if not isinstance(value, dict):
        if not isinstance(value, (str, AddedToken)):
            raise ValueError("Tokenicer: special token must be a string or AddedToken dictionary")
        return value
    fields = dict(value)
    marker = fields.pop("__type", "AddedToken")
    if marker != "AddedToken" or not isinstance(fields.get("content"), str):
        raise ValueError("Tokenicer: AddedToken metadata requires string content")
    if set(fields) - {"content", "single_word", "lstrip", "rstrip", "normalized", "special"}:
        raise ValueError("Tokenicer: unknown AddedToken metadata fields")
    if any(not isinstance(v, bool) for k, v in fields.items() if k != "content"):
        raise ValueError("Tokenicer: AddedToken flags must be booleans")
    fields["special"] = True
    return AddedToken(**fields)


def tokenizer_special_token_overrides(path, **kwargs) -> dict:
    tokenizer_config = _tokenizer_metadata(path, "tokenizer_config.json", **kwargs)
    has_decoder = "added_tokens_decoder" in kwargs or "added_tokens_decoder" in tokenizer_config
    special_tokens_map = {} if has_decoder else _tokenizer_metadata(path, "special_tokens_map.json", **kwargs)
    overrides = {}
    # HF's legacy branch lets the special-token map override config declarations.
    for metadata, legacy in ((tokenizer_config, False), (special_tokens_map, True)):
        for key, value in metadata.items():
            if key in kwargs:
                continue
            if key in {"additional_special_tokens", "extra_special_tokens"}:
                alias = "extra_special_tokens" if key == "additional_special_tokens" else "additional_special_tokens"
                if alias in kwargs or (alias in overrides and not legacy):
                    continue
                if isinstance(value, list):
                    tokens = [_special_token(token) for token in value]
                    if legacy:
                        existing = overrides.pop(alias, overrides.get(key, []))
                        tokens = existing + [token for token in tokens if token not in existing]
                    overrides[key] = tokens
            elif key.endswith("_token") and isinstance(value, dict):
                overrides[key] = _special_token(value)
            elif legacy and key.endswith("_token"):
                overrides[key] = None if value is None else _special_token(value)

    tokenizer_class = tokenizer_config.get("tokenizer_class")
    if not isinstance(tokenizer_class, str) or tokenizer_class not in {"LlamaTokenizer", "LlamaTokenizerFast"}:
        return overrides

    defaults = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
    }
    for key, default_value in defaults.items():
        if tokenizer_config.get(key, None) == "":
            overrides.setdefault(key, default_value)

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
