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

import re
import unittest
import warnings

from parameterized import parameterized

from tokenicer import Tokenicer


class TestLoadWarnings(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "/monster/data/model/opt-125",
                False,
                DeprecationWarning,
                r"Deprecated in 0\.9\.0: BPE\.__init__ will not create from files anymore, try `BPE\.from_file` instead",
            ),
            (
                "/monster/data/model/Qwen3-Omni-30B-A3B-Instruct",
                False,
                DeprecationWarning,
                r"Deprecated in 0\.9\.0: BPE\.__init__ will not create from files anymore, try `BPE\.from_file` instead",
            ),
            (
                "/monster/data/model/LongCat-Flash-Chat",
                True,
                FutureWarning,
                r"`rope_config_validation` is deprecated and has been removed\..*",
            ),
            (
                "/monster/data/model/OpenCoder-8B-Instruct",
                True,
                SyntaxWarning,
                r"invalid escape sequence '\\p'",
            ),
        ]
    )
    def test_known_external_load_warnings_are_suppressed(
        self,
        model_path: str,
        trust_remote_code: bool,
        warning_category,
        message_pattern: str,
    ):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Tokenicer.load(model_path, trust_remote_code=trust_remote_code)

        matching_warnings = [
            warning
            for warning in caught
            if issubclass(warning.category, warning_category)
            and re.search(message_pattern, str(warning.message))
        ]
        self.assertEqual(
            matching_warnings,
            [],
            msg=f"Tokenicer.load leaked {warning_category.__name__} warnings for {model_path}: {matching_warnings}",
        )
