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

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from huggingface_hub.errors import StrictDataclassClassValidationError
from tokenicer import Tokenicer
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerFast


# A Llama-3/GPT-4-style byte-level BPE. Transformers' legacy `LlamaTokenizer` class cannot
# represent this tokenization: it rebuilds the backend with a hardcoded sentencepiece
# `Metaspace` pre-tokenizer, which drops the leading whitespace of every word. Checkpoints
# like `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` ship exactly this combination and declare a
# `*TokenizerFast` class in `tokenizer_config.json`.
LLAMA3_SPLIT_PATTERN = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

CORPUS = (
    "Hello world, this is a test.",
    "The quick brown fox jumps over the lazy dog.",
    "def add(a, b):\n    return a + b\n",
    "Numbers: 1234567 3.14159 2024-10-15",
    "Unicode: café naïve 日本語 中文 🚀",
    "leading and repeated whitespace",
    "the model quantizes weights to low bit precision",
)

PROBES = (
    "Hello world, this is a test.",
    "  leading and   repeated   whitespace ",
    "def add(a, b):\n    return a + b\n",
    "Unicode: café naïve 日本語 中文 🚀",
)


def _build_byte_level_tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(LLAMA3_SPLIT_PATTERN, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ]
    )
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        CORPUS,
        trainer=trainers.BpeTrainer(
            vocab_size=512,
            special_tokens=["<unk>", "<s>", "</s>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        ),
    )
    return tokenizer


def _write_checkpoint(target: str, backend_tokenizer: Tokenizer, tokenizer_class: str) -> None:
    os.makedirs(target, exist_ok=True)
    backend_tokenizer.save(os.path.join(target, "tokenizer.json"))
    with open(os.path.join(target, "tokenizer_config.json"), "w", encoding="utf-8") as handle:
        json.dump(
            {
                "tokenizer_class": tokenizer_class,
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "add_bos_token": True,
                "add_eos_token": False,
                "clean_up_tokenization_spaces": False,
                "model_max_length": 131072,
            },
            handle,
        )
    with open(os.path.join(target, "config.json"), "w", encoding="utf-8") as handle:
        json.dump({"model_type": "llama", "architectures": ["LlamaForCausalLM"]}, handle)


class TestSerializedTokenizerPreference(unittest.TestCase):
    """`Tokenicer.load()` must keep the tokenization the checkpoint ships in `tokenizer.json`.

    Each test builds its own checkpoint on disk (no network) and asserts which tokenizer wins:
    the serialized one, or the class `Transformers` resolves.
    """

    @classmethod
    def setUpClass(cls):
        cls.workdir = tempfile.mkdtemp(prefix="tokenicer-serialized-")
        backend_tokenizer = _build_byte_level_tokenizer()

        # `huggingface_hub` lays a cached repo out as `models--<org>--<repo>/snapshots/<revision>`.
        cls.cached_dir = os.path.join(
            cls.workdir, "models--deepseek-ai--deepseek-r1-distill-llama-demo", "snapshots", "0123456789abcdef"
        )
        _write_checkpoint(cls.cached_dir, backend_tokenizer, tokenizer_class="LlamaTokenizerFast")

        # The same checkpoint copied out of the cache, e.g. a quantized model that was saved
        # and reloaded from a plain directory.
        cls.plain_dir = os.path.join(cls.workdir, "plain")
        _write_checkpoint(cls.plain_dir, backend_tokenizer, tokenizer_class="LlamaTokenizerFast")

        # A checkpoint that declares the legacy class keeps whatever Transformers resolves.
        cls.legacy_dir = os.path.join(cls.workdir, "legacy")
        _write_checkpoint(cls.legacy_dir, backend_tokenizer, tokenizer_class="LlamaTokenizer")

        # A checkpoint whose serialized tokenizer already matches the class it resolves to.
        cls.matching_dir = os.path.join(cls.workdir, "matching")
        AutoTokenizer.from_pretrained(cls.plain_dir, local_files_only=True).save_pretrained(cls.matching_dir)
        matching_config_path = os.path.join(cls.matching_dir, "tokenizer_config.json")
        with open(matching_config_path, "r", encoding="utf-8") as handle:
            matching_config = json.load(handle)
        matching_config["tokenizer_class"] = "LlamaTokenizerFast"
        with open(matching_config_path, "w", encoding="utf-8") as handle:
            json.dump(matching_config, handle)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.workdir, ignore_errors=True)

    @staticmethod
    def _probe_ids(tokenizer):
        # Token ids of the sample texts: the same comparison Tokenicer performs internally.
        return [tokenizer(text)["input_ids"] for text in PROBES]

    def test_fixture_reproduces_the_legacy_class_override(self):
        # The fixture must still exercise the Transformers behavior this correction exists for.
        legacy = AutoTokenizer.from_pretrained(self.plain_dir, local_files_only=True)
        serialized = PreTrainedTokenizerFast.from_pretrained(self.plain_dir, local_files_only=True)

        self.assertEqual(type(serialized).__name__, "TokenizersBackend")
        if self._probe_ids(legacy) == self._probe_ids(serialized):
            self.skipTest(
                "installed Transformers no longer resolves this fixture to a legacy class that "
                "overrides the serialized tokenizer; the fixture needs updating"
            )

    def test_cached_snapshot_uses_the_serialized_tokenizer(self):
        serialized = PreTrainedTokenizerFast.from_pretrained(self.cached_dir, local_files_only=True)

        tokenizer = Tokenicer.load(self.cached_dir)

        self.assertEqual(type(tokenizer.tokenizer).__name__, "TokenizersBackend")
        self.assertEqual(self._probe_ids(tokenizer), self._probe_ids(serialized))

    def test_plain_directory_uses_the_serialized_tokenizer(self):
        serialized = PreTrainedTokenizerFast.from_pretrained(self.plain_dir, local_files_only=True)

        tokenizer = Tokenicer.load(self.plain_dir)

        self.assertEqual(type(tokenizer.tokenizer).__name__, "TokenizersBackend")
        self.assertEqual(self._probe_ids(tokenizer), self._probe_ids(serialized))

    def test_explicit_opt_out_keeps_the_loaded_class(self):
        legacy = AutoTokenizer.from_pretrained(self.plain_dir, local_files_only=True)

        tokenizer = Tokenicer.load(self.plain_dir, prefer_serialized_tokenizer=False)

        self.assertEqual(type(tokenizer.tokenizer).__name__, "LlamaTokenizer")
        self.assertEqual(self._probe_ids(tokenizer), self._probe_ids(legacy))

    def test_legacy_class_declaration_is_preserved(self):
        legacy = AutoTokenizer.from_pretrained(self.legacy_dir, local_files_only=True)

        tokenizer = Tokenicer.load(self.legacy_dir)

        self.assertEqual(type(tokenizer.tokenizer).__name__, "LlamaTokenizer")
        self.assertEqual(self._probe_ids(tokenizer), self._probe_ids(legacy))

    def test_class_matching_the_serialized_tokenizer_is_preserved(self):
        loaded = AutoTokenizer.from_pretrained(self.matching_dir, local_files_only=True)

        tokenizer = Tokenicer.load(self.matching_dir)

        self.assertEqual(type(tokenizer.tokenizer).__name__, "LlamaTokenizer")
        self.assertEqual(self._probe_ids(tokenizer), self._probe_ids(loaded))

    def test_fallback_path_uses_the_serialized_tokenizer(self):
        serialized = PreTrainedTokenizerFast.from_pretrained(self.plain_dir, local_files_only=True)

        with patch(
            "tokenicer.tokenicer.AutoTokenizer.from_pretrained",
            side_effect=StrictDataclassClassValidationError(
                validator="validate_layer_type",
                cause=ValueError("legacy model config"),
            ),
        ):
            tokenizer = Tokenicer.load(self.plain_dir)

        self.assertEqual(type(tokenizer.tokenizer).__name__, "TokenizersBackend")
        self.assertEqual(self._probe_ids(tokenizer), self._probe_ids(serialized))


if __name__ == "__main__":
    unittest.main()
