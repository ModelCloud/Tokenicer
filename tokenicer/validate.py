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
import json
from transformers import PreTrainedTokenizerBase
from typing import Union, Optional

from .util import config_path, all_special_characters, isfile
from .const import VERIFY_JSON_FILE_NAME, VERIFY_ENCODE_PARAMS, VERIFY_DATASETS
from .config import VerifyData, VerifyConfig
from .exception import (
    VerificationFileNotFoundError,
    VerificationInitializationError,
    ChatTemplateError,
    ModelCofnfigNotFoundError
)


def _verify_file_exist(tokenizer):
    path = config_path(tokenizer)
    if path is None:
        raise ModelCofnfigNotFoundError()

    verify_json_path = os.path.join(path, VERIFY_JSON_FILE_NAME)
    return isfile(verify_json_path), verify_json_path


def _save(
        save_directory: Union[str, os.PathLike],
        tokenizer: PreTrainedTokenizerBase,
        use_chat_template: bool = True
    ):
    verify_json_path = os.path.join(save_directory, VERIFY_JSON_FILE_NAME)
    exist = isfile(verify_json_path)
    if exist:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("The verification file already exists.")
        return verify_json_path

    if use_chat_template and tokenizer.chat_template is None:
        raise ChatTemplateError()

    VERIFY_DATASETS.append(all_special_characters())

    prompts = []
    if use_chat_template:
        for data in VERIFY_DATASETS:
            message = [{"role": "user", "content": data}]
            prompt = tokenizer.apply_chat_template(
                message, add_generation_prompt=False, tokenize=False
            ).rstrip()
            prompts.append(prompt)
    else:
        prompts = VERIFY_DATASETS

    results = []
    for prompt in prompts:
        tokenized = tokenizer.encode_plus(prompt, **VERIFY_ENCODE_PARAMS)
        output = tokenized["input_ids"].tolist()[0]
        data = VerifyData(input=prompt, output=output)
        results.append(data)

    verify_dic = VerifyConfig(datasets=results).to_dict()

    with open(verify_json_path, 'w', encoding='utf-8') as f:
        json.dump(verify_dic, f, indent=4)
        f.write('\n')
    return verify_json_path


def _verify(tokenizer: PreTrainedTokenizerBase, verify_file_path: Optional[Union[str, os.PathLike]] = None) -> bool:
    exist = False

    if verify_file_path is not None:
        exist = isfile(verify_file_path)

    if not exist:
        exist, verify_json_path = _verify_file_exist(tokenizer)
        if not exist:
            raise VerificationFileNotFoundError()
    else:
        verify_json_path = verify_file_path

    with open(verify_json_path, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())

    config = VerifyConfig.from_dict(data)

    if config is None or len(config.datasets) == 0:
        raise VerificationInitializationError(verify_json_path=verify_json_path)

    for verify_data in config.datasets:
        input = verify_data.input
        tokenized = tokenizer.encode_plus(input, **VERIFY_ENCODE_PARAMS)["input_ids"].tolist()[0]
        if verify_data.output != tokenized:
            return False

    return True
