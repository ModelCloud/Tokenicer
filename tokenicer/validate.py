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
from .const import VALIDATE_JSON_FILE_NAME, VALIDATE_ENCODE_PARAMS, VALIDATE_DATASETS
from .config import ValidateConfig, ValidateData


def _validate_file_exist(tokenizer):
    path = config_path(tokenizer)
    if path is None:
        raise ValueError("Can not retrieve config path from the provided `pretrained_model_name_or_path`.")

    validate_json_path = os.path.join(path, VALIDATE_JSON_FILE_NAME)
    return isfile(validate_json_path), validate_json_path


def _save(
        save_dir: Union[str, os.PathLike],
        tokenizer: PreTrainedTokenizerBase,
        use_chat_template: bool = True
    ):
    os.makedirs(save_dir, exist_ok=True)

    validate_json_path = os.path.join(save_dir, VALIDATE_JSON_FILE_NAME)
    exist = isfile(validate_json_path)
    if exist:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Validate file:{validate_json_path} already exists.")
        return validate_json_path

    if use_chat_template and tokenizer.chat_template is None:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("Tokenizer does not support chat template.")
        use_chat_template = False

    VALIDATE_DATASETS.append(all_special_characters())

    prompts = []
    if use_chat_template:
        for data in VALIDATE_DATASETS:
            message = [{"role": "user", "content": data}]
            prompt = tokenizer.apply_chat_template(
                message, add_generation_prompt=False, tokenize=False
            ).rstrip()
            prompts.append(prompt)
    else:
        prompts = VALIDATE_DATASETS

    results = []
    for prompt in prompts:
        tokenized = tokenizer.encode_plus(prompt, **VALIDATE_ENCODE_PARAMS)
        output = tokenized["input_ids"].tolist()[0]
        data = ValidateData(input=prompt, output=output)
        results.append(data)

    validate_dic = ValidateConfig(data=results).to_dict()

    with open(validate_json_path, 'w', encoding='utf-8') as f:
        json.dump(validate_dic, f, indent=4)
        f.write('\n')
    return validate_json_path


def _validate(tokenizer: PreTrainedTokenizerBase, save_dir: Optional[Union[str, os.PathLike]] = None) -> bool:
    exist = False

    if save_dir is not None:
        validate_json_path = os.path.join(save_dir, VALIDATE_JSON_FILE_NAME)
        exist = isfile(validate_json_path)

    if not exist:
        exist, validate_json_path = _validate_file_exist(tokenizer)
        if not exist:
            raise ValueError("Validate file does not exist, please call the `save()` API first.")

    with open(validate_json_path, 'r', encoding='utf-8') as f:
        data = json.loads(f.read())

    config = ValidateConfig.from_dict(data)

    if config is None or len(config.data) == 0:
        raise ValueError(f"Init validate data failed, please check {validate_json_path}.")

    for data in config.data:
        input = data.input
        tokenized = tokenizer.encode_plus(input, **VALIDATE_ENCODE_PARAMS)["input_ids"].tolist()[0]
        if data.output != tokenized:
            return False

    return True
