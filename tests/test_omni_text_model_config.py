import unittest
import types
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tokenicer import Tokenicer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast


class DummyTokenizer:
    def __init__(self):
        self.vocab = {
            "<bos>": 0,
            "<eos>": 1,
            "<|fim_pad|>": 151662,
        }
        self.bos_token = "<bos>"
        self.bos_token_id = 0
        self.eos_token = "<eos>"
        self.eos_token_id = 1
        self.pad_token = None
        self.pad_token_id = None

    def get_vocab(self):
        return self.vocab

    def decode(self, token_ids):
        token_by_id = {token_id: token for token, token_id in self.vocab.items()}
        return "".join(token_by_id[token_id] for token_id in token_ids)


class DummyTextConfig:
    def __init__(self):
        self.model_type = "qwen2"
        self.bos_token_id = None
        self.bos_token = None
        self.eos_token_id = None
        self.eos_token = None
        self.pad_token_id = None


class DummyThinkerConfig:
    def __init__(self):
        self.text_config = DummyTextConfig()


class DummyOmniConfigWithGetTextConfig:
    def __init__(self):
        self.model_type = "qwen2_5_omni"
        self.thinker_config = DummyThinkerConfig()

    def get_text_config(self):
        return self.thinker_config.text_config


class DummyLegacyOmniConfigWithThinkerConfig:
    def __init__(self):
        self.model_type = "qwen2_5_omni"
        self.thinker_config = DummyThinkerConfig()


class DummyLegacyOmniConfigWithThinker:
    def __init__(self):
        self.model_type = "qwen2_5_omni"
        self.thinker = DummyThinkerConfig()


class TestOmniTextModelConfig(unittest.TestCase):
    def _assert_pad_token_fixed(self, model_config):
        tokenicer = Tokenicer()
        tokenicer.tokenizer = DummyTokenizer()
        tokenicer.model_config = model_config

        tokenicer.auto_fix_pad_token()

        resolved_model_config = Tokenicer._resolve_text_model_config(model_config)

        self.assertEqual(tokenicer.tokenizer.pad_token, "<|fim_pad|>")
        self.assertEqual(tokenicer.tokenizer.pad_token_id, 151662)
        self.assertEqual(resolved_model_config.bos_token_id, 0)
        self.assertEqual(resolved_model_config.eos_token_id, 1)

    def test_auto_fix_pad_token_uses_get_text_config(self):
        self._assert_pad_token_fixed(DummyOmniConfigWithGetTextConfig())

    def test_auto_fix_pad_token_uses_thinker_config_fallback(self):
        self._assert_pad_token_fixed(DummyLegacyOmniConfigWithThinkerConfig())

    def test_auto_fix_pad_token_uses_thinker_fallback(self):
        self._assert_pad_token_fixed(DummyLegacyOmniConfigWithThinker())

    def test_load_uses_explicit_model_config_for_composite_configs(self):
        backend = Tokenizer(WordLevel({"<pad>": 0, "<eos>": 1, "hello": 2}, unk_token="<pad>"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>")

        text_config = types.SimpleNamespace(
            model_type="qwen2_5_omni_text",
            pad_token_id=None,
            bos_token_id=None,
            eos_token_id=None,
        )

        class CompositeConfig:
            def get_text_config(self):
                return text_config

        wrapped = Tokenicer.load(tokenizer, model_config=CompositeConfig())

        self.assertIs(wrapped.model_config, text_config)
        self.assertEqual(text_config.pad_token_id, tokenizer.pad_token_id)
        self.assertEqual(text_config.eos_token_id, tokenizer.eos_token_id)

    def test_load_prefers_explicit_model_config_over_auto_config_for_string_paths(self):
        backend = Tokenizer(WordLevel({"<pad>": 0, "<eos>": 1, "hello": 2}, unk_token="<pad>"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>")

        text_config = types.SimpleNamespace(
            model_type="qwen2_5_omni_text",
            pad_token_id=None,
            bos_token_id=None,
            eos_token_id=None,
        )

        class CompositeConfig:
            def get_text_config(self):
                return text_config

        with patch.object(Tokenicer, "_load_tokenizer", return_value=tokenizer):
            with patch("tokenicer.tokenicer.auto_config", side_effect=AssertionError("auto_config should not run")):
                wrapped = Tokenicer.load("/tmp/fake-model", model_config=CompositeConfig())

        self.assertIs(wrapped.model_config, text_config)
        self.assertEqual(text_config.pad_token_id, tokenizer.pad_token_id)
        self.assertEqual(text_config.eos_token_id, tokenizer.eos_token_id)
