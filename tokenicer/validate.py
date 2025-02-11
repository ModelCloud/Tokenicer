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
from typing import Union, List
from .util import config_path
from .const import VERIFY_JSON_FILE_NAME, VERIFY_ENCODE_PARAMS, INPUT_KEY, TENSOR_KEY


def _verify_file_exist(tokenizer):
    path = config_path(tokenizer)
    if path is None:
        raise ValueError("Can not retrieve config path from the provided `pretrained_model_name_or_path`.")

    verify_json_path = os.path.join(path, VERIFY_JSON_FILE_NAME)

    if os.path.isfile(verify_json_path):
        return True, verify_json_path
    return False, verify_json_path


def _save_verify(prompts: Union[str, List[str]], tokenizer):
    exist, verify_json_path = _verify_file_exist(tokenizer)
    if exist:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning("The verification file already exists.")
        return verify_json_path

    if prompts is None:
        raise ValueError("`prompts` cannot be None")

    if not isinstance(prompts, str) and not isinstance(prompts, list):
        raise ValueError(
            f"Unsupported `prompts` type: Expected `str` or `List[str]`, actual = `{type(prompts)}`.")

    if isinstance(prompts, str):
        prompts = [prompts]

    if len(prompts) == 0:
        raise ValueError("len(prompts) == 0, `prompts` must be greater than 0")

    results = []
    for prompt in prompts:
        tokenized = tokenizer.encode_plus(prompt, **VERIFY_ENCODE_PARAMS)
        jsonl = {INPUT_KEY: prompt, TENSOR_KEY: tokenized["input_ids"].tolist()}
        results.append(jsonl)

    with open(verify_json_path, 'w') as f:
        for item in results:
            json.dump(item, f)
            f.write('\n')
    return verify_json_path


def _verify(tokenizer) -> bool:
    exist, verify_json_path = _verify_file_exist(tokenizer)
    if not exist:
        raise ValueError(f"The verification file does not exist, please call the `save_verify` API first")

    with open(verify_json_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_obj = json.loads(line)

            input_text = json_obj[INPUT_KEY]
            tensor = json_obj[TENSOR_KEY]

            tokenized = tokenizer.encode_plus(input_text, **VERIFY_ENCODE_PARAMS)
            if tensor != tokenized["input_ids"].tolist():
                return False
    return True
