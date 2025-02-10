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
from typing import Union, List, Optional
from transformers import PreTrainedTokenizerBase, PreTrainedModel, AutoTokenizer
from .util import candidate_id, config_path, auto_config
from .const import DEFAULT_PAD_TOKENS, MODEL_PAD_TOKEN_MAP

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Tokenicer:
    tokenizer: Union[str, PreTrainedTokenizerBase] = None
    trust_remote: bool = False
    model_config = None

    @classmethod
    def load(cls, tokenizer_or_path: Union[str, PreTrainedTokenizerBase], trust_remote: bool = False):
        if tokenizer_or_path is None:
            raise ValueError("`tokenizer_or_path` cannot be `None`.")
        tokenicer = cls()
        tokenicer.trust_remote = trust_remote

        path = None
        if isinstance(tokenizer_or_path, PreTrainedTokenizerBase):
            tokenizer = tokenizer_or_path
            tokenicer.tokenizer = tokenizer
            path = config_path(tokenizer)
        elif isinstance(tokenizer_or_path, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_path, trust_remote_code=trust_remote)
            if isinstance(tokenizer, PreTrainedTokenizerBase):
                tokenicer.tokenizer = tokenizer
                path = tokenizer_or_path
            else:
                ValueError(
                    f"Failed to initialize `tokenizer`: please ensure that the `tokenizer_or_path` parameter is set correctly.")
        else:
            raise ValueError(
                f"Unsupported `tokenizer_or_path` type: Expected `str` or `PreTrainedTokenizerBase`, actual = `{type(tokenizer_or_path)}`.")

        tokenicer.model_config = auto_config(path, trust_remote)

        if tokenicer.model_config is None:
            logger.warning(
                f"Auto model config retrieval from `tokenizer_or_path` failed. "
                f"Please pass a valid `model_or_path` argument to `auto_assign_pad_token()`.",
            )

        return tokenicer

    def auto_assign_pad_token(
        self,
        model_or_path: Optional[Union[str, PreTrainedModel]] = None,
        pad_tokens: Optional[List[Union[str, int]]] = None,
    ):
        model_config = None
        if model_or_path is not None:
            if isinstance(model_or_path, str):
                model_config = auto_config(model_or_path, self.trust_remote)
            elif isinstance(model_or_path, PreTrainedModel):
                model_config = getattr(model_or_path, "config", None)
            else:
                raise ValueError(
                    f"Unsupported `model_or_path` type: Expected `str` or `PreTrainedModel`, actual = `{type(model_or_path)}`.")

            if model_config is None:
                raise ValueError("Can not retrieve config from the provided `model_or_path`.")
        else:
            if self.model_config is not None:
                model_config = self.model_config
            else:
                raise ValueError(
                    f"Auto model config retrieval from `tokenizer_or_path` failed. "
                    f"Please pass a valid `model_or_path` argument to `auto_assign_pad_token()`.",
            )

        pad_token_id = model_config.pad_token_id

        if pad_token_id is None or pad_token_id in [model_config.bos_token_id, model_config.eos_token_id]:
            pad_token_id = self._auto_map_pad_token(model_config=model_config, pad_tokens=pad_tokens)
            if pad_token_id is None:
                raise ValueError(
                    "Model tokenizer requires fixing but we are unnable to auto-fix `pad_token`. Please consult model docks manually pass a `pad_token` to `load()`."
                )

        self.tokenizer.pad_token_id = pad_token_id
        self.tokenizer.pad_token = self.tokenizer.decode([pad_token_id])

        logger.info(f"Auto fixed pad_token_id={pad_token_id} (token='{self.tokenizer.pad_token}').")

    def _auto_map_pad_token(self, model_config, pad_tokens) -> Optional[int]:
        pad_token_id = None

        vocab = self.tokenizer.get_vocab()

        # Prioritize matching of pad token entered by the user
        if pad_tokens is not None:
            pad_token_id = candidate_id(pad_tokens, vocab)

        # Match MODEL_PAD_TOKEN_MAP to get pad token
        if pad_token_id is None and MODEL_PAD_TOKEN_MAP.get(model_config.model_type, None) is not None:
            tuple = MODEL_PAD_TOKEN_MAP.get(model_config.model_type)
            pad_token = tuple.token
            token_id = vocab.get(pad_token, None)
            if token_id is not None and token_id == tuple.token_id:
                pad_token_id = token_id

        # Match DEFAULT_PAD_TOKENS to get pad token
        if pad_token_id is None:
            pad_token_id = candidate_id(DEFAULT_PAD_TOKENS, vocab)

        # Use eos_token as pad token
        if pad_token_id is None:
            if isinstance(model_config.eos_token_id, list) and model_config.eos_token_id:
                pad_token_id = model_config.eos_token_id[0]
            else:
                pad_token_id = model_config.eos_token_id
        return pad_token_id

    def __getattr__(self, name):
        if hasattr(self.tokenizer, name):
            return getattr(self.tokenizer, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __call__(self, data, **kwargs):
        return self.tokenizer(data, **kwargs)

