# Copyright 2026 ModelCloud.ai
# Copyright 2026 qubitium@modelcloud.ai
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

import unittest
from unittest.mock import Mock, patch

from huggingface_hub.errors import StrictDataclassClassValidationError

from tokenicer import Tokenicer


class TestMistralRegexNormalization(unittest.TestCase):
    @patch("tokenicer.tokenicer.AutoTokenizer.from_pretrained")
    def test_mistral_regex_fix_is_enabled_by_default(self, from_pretrained):
        expected = object()
        from_pretrained.return_value = expected

        tokenizer = Tokenicer._load_tokenizer("poolside/Laguna-S-2.1", trust_remote_code=True)

        self.assertIs(tokenizer, expected)
        from_pretrained.assert_called_once_with(
            "poolside/Laguna-S-2.1",
            trust_remote_code=True,
            fix_mistral_regex=True,
        )

    @patch("tokenicer.tokenicer.AutoTokenizer.from_pretrained")
    def test_explicit_mistral_regex_setting_is_preserved(self, from_pretrained):
        from_pretrained.return_value = object()

        Tokenicer._load_tokenizer("poolside/Laguna-S-2.1", fix_mistral_regex=False)

        from_pretrained.assert_called_once_with(
            "poolside/Laguna-S-2.1",
            fix_mistral_regex=False,
        )

    @patch("tokenicer.tokenicer.Tokenicer._resolve_tokenizer_class")
    @patch("tokenicer.tokenicer.tokenizer_class_name", return_value="FallbackTokenizer")
    @patch("tokenicer.tokenicer.tokenizer_special_token_overrides", return_value={"bos_token": "<s>"})
    @patch(
        "tokenicer.tokenicer.AutoTokenizer.from_pretrained",
        side_effect=StrictDataclassClassValidationError(
            validator="validate_layer_type",
            cause=ValueError("legacy model config"),
        ),
    )
    def test_mistral_regex_fix_is_preserved_on_fallback(
        self,
        auto_from_pretrained,
        special_token_overrides,
        tokenizer_class_name,
        resolve_tokenizer_class,
    ):
        fallback_from_pretrained = Mock(return_value=object())
        resolve_tokenizer_class.return_value = Mock(from_pretrained=fallback_from_pretrained)

        Tokenicer._load_tokenizer("/tmp/Laguna-S-2.1", trust_remote_code=True)

        auto_from_pretrained.assert_called_once_with(
            "/tmp/Laguna-S-2.1",
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
        special_token_overrides.assert_called_once_with("/tmp/Laguna-S-2.1")
        tokenizer_class_name.assert_called_once_with("/tmp/Laguna-S-2.1")
        resolve_tokenizer_class.assert_called_once_with("/tmp/Laguna-S-2.1", "FallbackTokenizer")
        fallback_from_pretrained.assert_called_once_with(
            "/tmp/Laguna-S-2.1",
            trust_remote_code=True,
            fix_mistral_regex=True,
            bos_token="<s>",
        )


if __name__ == "__main__":
    unittest.main()
