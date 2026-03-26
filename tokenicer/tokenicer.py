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

import logging
import os
import warnings
from importlib import import_module
from typing import Any, List, Optional, Union

from transformers import (
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module

try:
    from huggingface_hub.errors import StrictDataclassFieldValidationError
except Exception:  # pragma: no cover - optional dependency path
    StrictDataclassFieldValidationError = None

from .const import DEFAULT_PAD_TOKENS, MODEL_PAD_TOKEN_MAP
from .util import (
    auto_config,
    config_path,
    custom_tokenizer_class_ref,
    has_custom_tokenizer_code,
    tokenizer_class_name,
    tokenizer_special_token_overrides,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_TOKENIZER_LOAD_EXCEPTIONS = (
    AttributeError,
    OSError,
    RecursionError,
    TypeError,
    ValueError,
    KeyError,
)

if StrictDataclassFieldValidationError is not None:
    _TOKENIZER_LOAD_EXCEPTIONS = _TOKENIZER_LOAD_EXCEPTIONS + (StrictDataclassFieldValidationError,)

_KNOWN_LOAD_WARNING_SUPPRESSIONS = [
    (
        ("opt-125",),
        DeprecationWarning,
        r"Deprecated in 0\.9\.0: BPE\.__init__ will not create from files anymore, try `BPE\.from_file` instead",
    ),
    (
        ("Qwen3-Omni-30B-A3B-Instruct",),
        DeprecationWarning,
        r"Deprecated in 0\.9\.0: BPE\.__init__ will not create from files anymore, try `BPE\.from_file` instead",
    ),
    (
        ("LongCat-Flash-Chat",),
        FutureWarning,
        r"`rope_config_validation` is deprecated and has been removed\..*",
    ),
    (
        ("OpenCoder-8B-Instruct",),
        SyntaxWarning,
        "invalid escape sequence '\\\\p'",
    ),
]


class Tokenicer():

    def __init__(self):
        pass

    _compatibility_shims_installed = False

    @classmethod
    def load(
        cls,
        pretrained_model_name_or_path: Union[str, PreTrainedTokenizerBase],
        strict: bool = False,
        pad_tokens: Optional[List[Union[str, int]]] = None,
        model_config: Any = None,
        **kwargs,
    ):
        if pretrained_model_name_or_path is None:
            raise ValueError("Tokenicer: `pretrained_model_name_or_path` cannot be `None`.")

        trust_remote_code = kwargs.get('trust_remote_code', False)

        warning_filters = cls._warning_filters(pretrained_model_name_or_path)
        with warnings.catch_warnings():
            for category, message in warning_filters:
                warnings.filterwarnings("ignore", category=category, message=message)

            if isinstance(pretrained_model_name_or_path, PreTrainedTokenizerBase):
                tokenizer = pretrained_model_name_or_path
                path = config_path(tokenizer)
            elif isinstance(pretrained_model_name_or_path, str):
                tokenizer = cls._load_tokenizer(pretrained_model_name_or_path, **kwargs)
                if isinstance(tokenizer, PreTrainedTokenizerBase):
                    path = pretrained_model_name_or_path
                else:
                    raise ValueError("Tokenicer: Failed to initialize `tokenizer`: please ensure the `pretrained_model_name_or_path` is set correctly.")
            else:
                raise ValueError(f"Tokenicer: Unsupported `pretrained_model_name_or_path` type: Expected `str` or `PreTrainedTokenizerBase`, actual = `{type(pretrained_model_name_or_path)}`.")

            if model_config is None:
                model_config = auto_config(path, trust_remote_code)

            resolved_model_config = cls._resolve_text_model_config(model_config)

        if resolved_model_config is None:
            logger.warning(
                "Tokenicer: Auto model config retrieval from `pretrained_model_name_or_path` failed. "
                "Please pass `model_config=` to `load()` or a valid `model_or_path` argument to `auto_assign_pad_token()`.",
            )

        # dynamically change Tokenicer's type to tokenizer's
        tokenizer_cls = type(tokenizer)
        tokenicer_cls_wrapper = type(f"{tokenizer_cls.__name__}", (cls, tokenizer_cls), {})

        t = tokenicer_cls_wrapper()
        t.tokenizer = tokenizer
        t.model_config = resolved_model_config
        t.auto_fix_pad_token(strict=strict, pad_tokens=pad_tokens)
        return t

    @staticmethod
    def _load_tokenizer(pretrained_model_name_or_path: str, **kwargs):
        Tokenicer._install_tokenizer_compatibility_shims()
        try:
            # Keep the normal Transformers path first so standard checkpoints behave unchanged.
            return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
        except _TOKENIZER_LOAD_EXCEPTIONS:
            overrides = tokenizer_special_token_overrides(pretrained_model_name_or_path)
            retry_kwargs = dict(kwargs)
            retry_kwargs.update(overrides)

            tokenizer_cls_name = tokenizer_class_name(pretrained_model_name_or_path)
            if tokenizer_cls_name is not None:
                # Some checkpoints have a valid tokenizer_config.json even when config auto-detection is broken.
                tokenizer_cls = Tokenicer._resolve_tokenizer_class(
                    pretrained_model_name_or_path,
                    tokenizer_cls_name,
                )
                if tokenizer_cls is not None:
                    logger.warning(
                        "Tokenicer: Retrying tokenizer load via `%s.from_pretrained()` for `%s`.",
                        tokenizer_cls_name,
                        pretrained_model_name_or_path,
                    )
                    return tokenizer_cls.from_pretrained(pretrained_model_name_or_path, **retry_kwargs)

            custom_class_ref = custom_tokenizer_class_ref(pretrained_model_name_or_path)
            if custom_class_ref is not None:
                # Load the declared custom tokenizer class directly to bypass bad fallback config parsing.
                try:
                    logger.warning(
                        "Tokenicer: Retrying tokenizer load via dynamic tokenizer class `%s` for `%s`.",
                        custom_class_ref,
                        pretrained_model_name_or_path,
                    )
                    tokenizer_cls = get_class_from_dynamic_module(custom_class_ref, pretrained_model_name_or_path)
                    retry_kwargs.pop("trust_remote_code", None)
                    return tokenizer_cls.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True, **retry_kwargs)
                except Exception:
                    logger.warning(
                        "Tokenicer: Failed custom tokenizer load via `%s` for `%s`, trying local tokenizer files.",
                        custom_class_ref,
                        pretrained_model_name_or_path,
                    )

            if kwargs.get("trust_remote_code", False) or not has_custom_tokenizer_code(pretrained_model_name_or_path):
                raise

            retry_kwargs = dict(kwargs)
            retry_kwargs["trust_remote_code"] = True
            # Local checkpoints with custom tokenizer code can still succeed once remote code is explicitly allowed.
            logger.warning(
                "Tokenicer: Retrying tokenizer load with `trust_remote_code=True` for local custom tokenizer `%s`.",
                pretrained_model_name_or_path,
            )
            return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **retry_kwargs)

    @staticmethod
    def _warning_filters(pretrained_model_name_or_path):
        if not isinstance(pretrained_model_name_or_path, str):
            return []

        model_path = pretrained_model_name_or_path
        for model_parts, warning_category, pattern in _KNOWN_LOAD_WARNING_SUPPRESSIONS:
            if any(part in model_path for part in model_parts):
                yield warning_category, pattern

    @staticmethod
    def _install_tokenizer_compatibility_shims():
        if Tokenicer._compatibility_shims_installed:
            return

        Tokenicer._compatibility_shims_installed = True

        # Kimi tokenizer snapshots may import bytes_to_unicode from a module path
        # that has moved in newer transformers.
        try:
            from transformers.convert_slow_tokenizer import bytes_to_unicode
            from transformers.models.gpt2 import tokenization_gpt2

            if not hasattr(tokenization_gpt2, "bytes_to_unicode"):
                tokenization_gpt2.bytes_to_unicode = bytes_to_unicode
        except Exception:
            return

    @staticmethod
    def _resolve_tokenizer_class(
        pretrained_model_name_or_path: str,
        tokenizer_cls_name: str,
    ):
        import transformers

        if tokenizer_cls_name == "PreTrainedTokenizer":
            return getattr(transformers, "PreTrainedTokenizerFast", None)

        tokenizer_cls = getattr(transformers, tokenizer_cls_name, None)
        if tokenizer_cls is not None:
            # Some checkpoints only expose a base tokenizer marker in
            # tokenizer_config.json. Prefer the fast implementation.
            return tokenizer_cls

        # Handle explicit module references in config (e.g. tokenization_kimi.Tokenizer).
        if "." in tokenizer_cls_name:
            module_name, class_name = tokenizer_cls_name.rsplit(".", 1)
            try:
                module = import_module(module_name)
                tokenizer_cls = getattr(module, class_name, None)
            except Exception:
                tokenizer_cls = None
            if tokenizer_cls is not None:
                return tokenizer_cls

        # Fallback scan local tokenizer files for custom class names.
        if not isinstance(pretrained_model_name_or_path, str) or not os.path.isdir(pretrained_model_name_or_path):
            return None

        local_modules = [
            f[:-3]
            for f in os.listdir(pretrained_model_name_or_path)
            if f.endswith(".py") and (f.startswith("tokenization_") or f in {"tiktoken.py"})
        ]

        for module_name in local_modules:
            try:
                tokenizer_cls = get_class_from_dynamic_module(
                    f"{module_name}.{tokenizer_cls_name}",
                    pretrained_model_name_or_path,
                )
                return tokenizer_cls
            except Exception:
                pass

        return None

    def auto_fix_pad_token(
        self,
        model_or_path: Optional[Union[str, PreTrainedModel]] = None,
        pad_tokens: Optional[List[Union[str, int]]] = None,
        strict: bool = False,
    ):
        if model_or_path is not None:
            if isinstance(model_or_path, str):
                model_config = auto_config(model_or_path, self.tokenizer.trust_remote_code)
            elif isinstance(model_or_path, PreTrainedModel):
                model_config = getattr(model_or_path, "config", None)
            else:
                raise ValueError(
                    f"Tokenicer: Unsupported `model_or_path` type: Expected `str` or `PreTrainedModel`, actual = `{type(model_or_path)}`.")

            if model_config is None:
                raise ValueError("Tokenicer: Can not retrieve config from the provided `model_or_path`.")
        else:
            if self.model_config is not None:
                model_config = self.model_config
            else:
                # Tokenizer-only bundles can still preserve an existing pad/eos setup without model config.
                pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
                if pad_token_id is not None:
                    if getattr(self.tokenizer, "pad_token", None) is None:
                        # Do not round-trip special token ids through decode().
                        # decode() returns rendered text, not the canonical token literal,
                        # and some custom tokenizers intentionally inject formatting.
                        # ERNIE 4.5 is the concrete regression here: decode([0]) returns
                        # " <unk>" while the real special token is "<unk>".
                        self.tokenizer.pad_token = self._token_literal_for_id(pad_token_id)
                    return

                if not strict and getattr(self.tokenizer, "eos_token_id", None) is not None:
                    self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    logger.warning(
                        "Tokenicer: No model config found; falling back to tokenizer.eos_token as pad_token."
                    )
                    return

                raise ValueError(
                    "Tokenicer: Auto model config retrieval from `pretrained_model_name_or_path` failed. "
                    "Please pass `model_config=` to `load()` or a valid `model_or_path` argument to `auto_assign_pad_token()`.",
            )

        model_config = self._resolve_text_model_config(model_config)
        self.model_config = model_config

        self.auto_fix_model_config(model_config)

        pad_token_id = getattr(model_config, "pad_token_id", None)
        has_invalid_config_pad = (
            hasattr(model_config, "bos_token_id")
            and hasattr(model_config, "eos_token_id")
            and pad_token_id in [model_config.bos_token_id, model_config.eos_token_id]
        )

        # Explicit pad token candidates should always be able to override a
        # previously synchronized config pad_token_id.
        if pad_tokens is not None or pad_token_id is None or has_invalid_config_pad:
            pad_token_id = self._auto_map_pad_token(model_config=model_config, pad_tokens=pad_tokens)

            if not strict:
                if pad_token_id is None and self.tokenizer.eos_token_id is not None:
                    pad_token_id = self.tokenizer.eos_token_id
                    logger.warning(
                        "Tokenicer: Auto model config unable to fix `pad_token`, Use tokenizer.eos_token as pad_token"
                        "pad_token = eos_token, There may be problems with the model during training or inference."
                        "It is recommended that you manually pass a `pad_tokens` to `load()`",
                    )

            if pad_token_id is None:
                raise ValueError(
                    "Tokenicer: Model tokenizer requires fixing but we are unable to auto-fix `pad_token`. Please consult model docs and pass `pad_tokens` to `load()` or set `strict`= False."
                )

        self.tokenizer.pad_token_id = pad_token_id
        # Preserve the canonical special-token spelling instead of using
        # decode([pad_token_id]). For tokenizers like ERNIE 4.5, decode() is a
        # detokenization API and may prepend whitespace to special tokens.
        self.tokenizer.pad_token = self._token_literal_for_id(pad_token_id)

        if getattr(model_config, "pad_token_id", None) is None:
            # Keep the resolved text config aligned so downstream callers do
            # not need their own tokenizer-to-config synchronization wrapper.
            model_config.pad_token_id = pad_token_id

        logger.info(f"Tokenicer: Auto fixed pad_token_id={pad_token_id} (token='{self.tokenizer.pad_token}').")

    def _token_literal_for_id(self, token_id: int) -> str:
        # Prefer declared special-token metadata over decode(). The metadata is
        # the canonical token literal used by save/load, while decode() is a
        # text-rendering path and may be lossy for standalone special tokens.
        special_token_attrs = (
            "pad_token",
            "unk_token",
            "eos_token",
            "bos_token",
            "cls_token",
            "sep_token",
            "mask_token",
        )

        for attr in special_token_attrs:
            attr_id = getattr(self.tokenizer, f"{attr}_id", None)
            attr_value = getattr(self.tokenizer, attr, None)
            if attr_id == token_id and attr_value is not None:
                return attr_value

        special_tokens_map = getattr(self.tokenizer, "special_tokens_map", {}) or {}
        for attr_value in special_tokens_map.values():
            if isinstance(attr_value, list):
                for item in attr_value:
                    if isinstance(item, str) and self.tokenizer.convert_tokens_to_ids(item) == token_id:
                        return item
            elif isinstance(attr_value, str) and self.tokenizer.convert_tokens_to_ids(attr_value) == token_id:
                return attr_value

        # Final fallback for tokenizers that do not expose enough special-token
        # metadata to recover the literal directly.
        return self.tokenizer.decode([token_id])

    @staticmethod
    def _resolve_text_model_config(model_config, seen=None):
        if model_config is None:
            return None

        if seen is None:
            seen = set()

        model_config_id = id(model_config)
        if model_config_id in seen:
            return model_config
        seen.add(model_config_id)

        get_text_config = getattr(model_config, "get_text_config", None)
        if callable(get_text_config):
            try:
                text_model_config = get_text_config()
            except (AttributeError, TypeError, ValueError):
                text_model_config = None
            if text_model_config is not None and text_model_config is not model_config:
                return Tokenicer._resolve_text_model_config(text_model_config, seen=seen)

        # Older multimodal configs may keep the text decoder under thinker/text_config rather than top-level.
        for attr_name in ("text_config", "thinker_config", "thinker"):
            nested_model_config = getattr(model_config, attr_name, None)
            if nested_model_config is None or nested_model_config is model_config:
                continue

            resolved_model_config = Tokenicer._resolve_text_model_config(nested_model_config, seen=seen)
            if resolved_model_config is not None:
                return resolved_model_config

        return model_config

    def _auto_map_pad_token(self, model_config, pad_tokens) -> Optional[int]:
        pad_token_id = None

        # Prioritize matching of pad token entered by the user
        if pad_tokens is not None:
            pad_token_id = self._candidate_id_from_tokenizer(pad_tokens)

        if pad_tokens is None and getattr(self.tokenizer, "pad_token_id", None) is not None:
            return self.tokenizer.pad_token_id

        # Match MODEL_PAD_TOKEN_MAP to get pad token
        model_type = getattr(model_config, "model_type", None)
        if pad_token_id is None and MODEL_PAD_TOKEN_MAP.get(model_type, None) is not None:
            token_tuple = MODEL_PAD_TOKEN_MAP.get(model_type)
            token_id = self._candidate_id_from_tokenizer([token_tuple.token])
            if token_id is not None and token_id == token_tuple.token_id:
                pad_token_id = token_id

        # Match DEFAULT_PAD_TOKENS to get pad token
        if pad_token_id is None:
            pad_token_id = self._candidate_id_from_tokenizer(DEFAULT_PAD_TOKENS)

        # Use eos_token as pad token
        if pad_token_id is None:
            eos_token_id = getattr(model_config, "eos_token_id", None)
            if isinstance(eos_token_id, list) and eos_token_id:
                pad_token_id = eos_token_id[0]
            else:
                pad_token_id = eos_token_id

        return pad_token_id

    def auto_fix_model_config(self, model_config):
        if getattr(model_config, "bos_token_id", None) is None and getattr(self.tokenizer, "bos_token_id", None) is not None:
            model_config.bos_token = self.tokenizer.bos_token
            model_config.bos_token_id = self.tokenizer.bos_token_id

        tokenizer_eos_token_id = self._tokenizer_eos_token_id()
        if getattr(model_config, "eos_token_id", None) is None and tokenizer_eos_token_id is not None:
            model_config.eos_token = self._token_literal_for_id(tokenizer_eos_token_id)
            model_config.eos_token_id = tokenizer_eos_token_id

    def _candidate_id_from_tokenizer(self, token_list: List[Union[str, int]]) -> Optional[int]:
        vocab = self.tokenizer.get_vocab()

        for item in token_list:
            if isinstance(item, str):
                token_id = vocab.get(item)
                if token_id is None:
                    token_id = self.tokenizer.convert_tokens_to_ids(item)
                if isinstance(token_id, int) and token_id >= 0:
                    return token_id
            elif isinstance(item, int) and 0 <= item < len(self.tokenizer):
                return item

        return None

    def _tokenizer_eos_token_id(self) -> Optional[int]:
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, list):
            return eos_token_id[0] if eos_token_id else None
        if eos_token_id is not None:
            return eos_token_id
        return getattr(self.tokenizer, "eod_id", None)

    def __getattribute__(self, name):
        if name in {
            "tokenizer",
        }:
            return super().__getattribute__(name)

        try:
            return super().__getattribute__("tokenizer").__getattribute__(name)
        except AttributeError:
            return super().__getattribute__(name)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __setattr__(self, name, value):
        if name in {"tokenizer", "model_config"}:
            return super().__setattr__(name, value)

        try:
            tokenizer = super().__getattribute__("tokenizer")
        except AttributeError:
            tokenizer = None
        if tokenizer is not None and hasattr(tokenizer, name):
            # Tokenicer proxies tokenizer reads through `self.tokenizer`, so mutable tokenizer
            # fields like `padding_side` must also write through to the wrapped tokenizer.
            return setattr(tokenizer, name, value)

        return super().__setattr__(name, value)

    def __call__(self, data, **kwargs):
        return self.tokenizer(data, **kwargs)
