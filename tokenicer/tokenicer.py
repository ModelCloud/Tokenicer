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

import os
import logging
from typing import Union, List, Optional, Tuple
from transformers import PreTrainedTokenizerBase, PreTrainedModel, AutoTokenizer
from .util import candidate_id, config_path, auto_config
from .const import DEFAULT_PAD_TOKENS, MODEL_PAD_TOKEN_MAP
from .validate import _save, _verify, _verify_file_exist

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Tokenicer:
    tokenizer: Union[str, PreTrainedTokenizerBase] = None
    model_config = None

    @classmethod
    def load(cls, pretrained_model_name_or_path: Union[str, PreTrainedTokenizerBase], strict: bool = False, pad_tokens: Optional[List[Union[str, int]]] = None, **kwargs):
        if pretrained_model_name_or_path is None:
            raise ValueError("`pretrained_model_name_or_path` cannot be `None`.")

        trust_remote_code = kwargs.get('trust_remote_code', False)

        tokenicer = cls()

        path = None
        if isinstance(pretrained_model_name_or_path, PreTrainedTokenizerBase):
            tokenizer = pretrained_model_name_or_path
            tokenicer.tokenizer = tokenizer
            path = config_path(tokenizer)
        elif isinstance(pretrained_model_name_or_path, str):
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
            if isinstance(tokenizer, PreTrainedTokenizerBase):
                tokenicer.tokenizer = tokenizer
                path = pretrained_model_name_or_path
            else:
                ValueError(
                    f"Failed to initialize `tokenizer`: please ensure that the `pretrained_model_name_or_path` parameter is set correctly.")
        else:
            raise ValueError(
                f"Unsupported `pretrained_model_name_or_path` type: Expected `str` or `PreTrainedTokenizerBase`, actual = `{type(pretrained_model_name_or_path)}`.")



        tokenicer.model_config = auto_config(path, trust_remote_code)

        if tokenicer.model_config is None:
            logger.warning(
                f"Auto model config retrieval from `pretrained_model_name_or_path` failed. "
                f"Please pass a valid `model_or_path` argument to `auto_assign_pad_token()`.",
            )

        tokenicer.auto_fix_pad_token(strict=strict, pad_tokens=pad_tokens)

        exist, _ = _verify_file_exist(tokenizer)
        if exist:
            tokenicer.verify()

        return tokenicer

    def auto_fix_pad_token(
        self,
        model_or_path: Optional[Union[str, PreTrainedModel]] = None,
        pad_tokens: Optional[List[Union[str, int]]] = None,
        strict: bool = False,
    ):
        model_config = None
        if model_or_path is not None:
            if isinstance(model_or_path, str):
                model_config = auto_config(model_or_path, self.tokenizer.trust_remote_code)
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
                    f"Auto model config retrieval from `pretrained_model_name_or_path` failed. "
                    f"Please pass a valid `model_or_path` argument to `auto_assign_pad_token()`.",
            )

        self.auto_fix_model_config(model_config)

        pad_token_id = model_config.pad_token_id

        if pad_token_id is None or pad_token_id in [model_config.bos_token_id, model_config.eos_token_id]:
            pad_token_id = self._auto_map_pad_token(model_config=model_config, pad_tokens=pad_tokens)

            if not strict:
                if pad_token_id is None and self.tokenizer.eos_token_id is not None:
                    pad_token_id = self.tokenizer.eos_token_id
                    logger.warning(
                        f"Auto model config unable to fix `pad_token`, Use tokenizer.eos_token as pad_token"
                        f"pad_token = eos_token, There may be problems with the model during training or inference."
                        f"It is recommended that you manually pass a `pad_tokens` to `load()`",
                    )

            if pad_token_id is None:
                raise ValueError(
                    "Model tokenizer requires fixing but we are unable to auto-fix `pad_token`. Please consult model docks manually pass a `pad_tokens` to `load()` or set `strict`= False."
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
            token_tuple = MODEL_PAD_TOKEN_MAP.get(model_config.model_type)
            pad_token = token_tuple.token
            token_id = vocab.get(pad_token, None)
            if token_id is not None and token_id == token_tuple.token_id:
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

    def auto_fix_model_config(self, model_config):
        if model_config.bos_token_id is None and self.tokenizer.bos_token_id is not None:
            model_config.bos_token = self.tokenizer.bos_token
            model_config.bos_token_id = self.tokenizer.bos_token_id

        if model_config.eos_token_id is None and self.tokenizer.eos_token_id is not None:
            model_config.eos_token = self.tokenizer.eos_token
            model_config.eos_token_id = self.tokenizer.eos_token_id

    def save(self, save_dir: Union[str, os.PathLike], use_chat_template: bool = True) -> str:
        return _save(save_dir=save_dir, tokenizer=self.tokenizer, use_chat_template=use_chat_template)

    def verify(self, save_dir: Union[str, os.PathLike] = None) -> bool:
        return _verify(self.tokenizer, save_dir=save_dir)

    def __getattr__(self, name):
        if hasattr(self.tokenizer, name):
            return getattr(self.tokenizer, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __call__(self, data, **kwargs):
        return self.tokenizer(data, **kwargs)

