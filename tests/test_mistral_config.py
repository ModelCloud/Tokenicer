# Copyright 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers
from transformers import AutoTokenizer, PreTrainedConfig, PreTrainedTokenizerFast

from tokenicer import Tokenicer


@pytest.fixture
def legacy_bpe(tmp_path):
    vocab = {token: i for i, token in enumerate(["<unk>", "<s>", "</s>", "▁", "H", "e", "l", "o"])}
    for value in range(256):
        vocab[f"<0x{value:02X}>"] = len(vocab)
    for value in range(100_001 - len(vocab)):
        vocab[f"unused{value}"] = len(vocab)
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>", byte_fallback=True))
    backend.normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
    backend.decoder = decoders.Sequence([
        decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse(), decoders.Strip(" ", 1, 0),
    ])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", bos_token="<s>", eos_token="</s>")
    tokenizer.save_pretrained(tmp_path)
    metadata = json.loads((tmp_path / "tokenizer_config.json").read_text())
    metadata.update(tokenizer_class="LlamaTokenizer", legacy=True)
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(metadata))
    (tmp_path / "config.json").write_text(json.dumps({"transformers_version": "4.36.0"}))
    return tmp_path, backend


@pytest.mark.parametrize("family", ["minicpm", "llama"])
@pytest.mark.parametrize("selector", ["config", "model_config", "both", "composite"])
def test_supplied_config_preserves_non_mistral_bpe(legacy_bpe, family, selector):
    path, backend = legacy_bpe
    config = PreTrainedConfig(model_type=family, pad_token_id=0)
    kwargs = {"config": config} if selector == "config" else {"model_config": config}
    if selector in {"both", "composite"}:
        kwargs["config"] = config
    if selector == "composite":
        kwargs["model_config"] = SimpleNamespace(text_config=config)
    text = "Hello GPT-QModel! 你好，世界。"
    expected = backend.encode(text, add_special_tokens=False).ids
    native = AutoTokenizer.from_pretrained(path, config=config, trust_remote_code=False)
    assert native.encode(text, add_special_tokens=False) == expected
    wrapped = Tokenicer.load(str(path), trust_remote_code=False, **kwargs)
    assert wrapped.encode(text, add_special_tokens=False) == expected
    assert wrapped.decode(expected) == native.decode(expected)


@pytest.mark.parametrize("family", ["mistral", "mistral3", "voxtral", "ministral", "pixtral", "laguna",
                                    "mixtral", "ministral3", "mistral_large4_text", "voxtral_realtime_text",
                                    None, "", 17])
def test_mistral_and_unknown_configs_keep_existing_default(family):
    config = SimpleNamespace(model_type=family)
    with patch("tokenicer.util.cached_file", return_value=None), \
            patch("tokenicer.tokenicer.AutoTokenizer.from_pretrained", return_value=object()) as load:
        Tokenicer._load_tokenizer("no-network", config=config, trust_remote_code=True)
    assert load.call_args.kwargs["fix_mistral_regex"] is True


@pytest.mark.parametrize("override", [False, True, None])
@pytest.mark.parametrize("family", ["minicpm", "mistral"])
def test_explicit_override_remains_authoritative(family, override):
    config = SimpleNamespace(model_type=family)
    with patch("tokenicer.util.cached_file", return_value=None), \
            patch("tokenicer.tokenicer.AutoTokenizer.from_pretrained", return_value=object()) as load:
        Tokenicer._load_tokenizer("no-network", config=config, fix_mistral_regex=override, trust_remote_code=True)
    assert load.call_args.kwargs["fix_mistral_regex"] is override


def test_exact_auto_config_takes_precedence_over_padding_model_config():
    with patch("tokenicer.util.cached_file", return_value=None), \
            patch("tokenicer.tokenicer.AutoTokenizer.from_pretrained", return_value=object()) as load:
        Tokenicer._load_tokenizer("no-network", config=SimpleNamespace(model_type="minicpm"),
                                  model_config=SimpleNamespace(model_type="mistral"), trust_remote_code=True)
    assert load.call_args.kwargs["fix_mistral_regex"] is False


@pytest.mark.parametrize("family, expected", [("minicpm", False), ("mistral", True)])
def test_default_is_preserved_by_fallback_and_retry(family, expected):
    fallback = Mock(from_pretrained=Mock(return_value=object()))
    with patch("tokenicer.tokenicer.AutoTokenizer.from_pretrained", side_effect=ValueError("retry")) as load, \
            patch("tokenicer.util.cached_file", return_value=None), \
            patch.object(Tokenicer, "_declared_tokenizer_class", return_value="LlamaTokenizer"), \
            patch("tokenicer.tokenicer.Tokenicer._resolve_tokenizer_class", return_value=fallback):
        Tokenicer._load_tokenizer("no-network", config=SimpleNamespace(model_type=family), trust_remote_code=True)
    assert load.call_args.kwargs["fix_mistral_regex"] is expected
    assert fallback.from_pretrained.call_args.kwargs["fix_mistral_regex"] is expected


def test_malformed_native_assets_fail_then_repaired_path_retries(legacy_bpe):
    path, backend = legacy_bpe
    config = PreTrainedConfig(model_type="minicpm", pad_token_id=0)
    serialized = (path / "tokenizer_config.json").read_bytes()
    (path / "tokenizer_config.json").write_text("{broken")
    with pytest.raises(json.JSONDecodeError):
        Tokenicer.load(str(path), config=config, model_config=config, trust_remote_code=False)
    (path / "tokenizer_config.json").write_bytes(serialized)
    wrapped = Tokenicer.load(str(path), config=config, model_config=config, trust_remote_code=False)
    text = "Hello 你好"
    assert wrapped.encode(text, add_special_tokens=False) == backend.encode(text, add_special_tokens=False).ids


@pytest.mark.parametrize("name, revision", [
    ("MiniCPM-2B-128k", "6011acb51acf3e8c0cb10c428d7064dc39431720"),
    ("MiniCPM-2B-dpo-bf16", "9d6de77274c9f364037bf8506b4a831d3520ae79"),
])
def test_pinned_original_minicpm_ids(name, revision, tmp_path):
    from huggingface_hub import snapshot_download

    source = snapshot_download(f"openbmb/{name}", revision=revision, allow_patterns=[
        "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    ])
    payload = json.loads((Path(source) / "config.json").read_text())
    assert "model_type" not in payload
    config = PreTrainedConfig(**payload, model_type="minicpm")
    text = "Hello GPT-QModel! 你好，世界。"
    expected = [21045, 1527, 7809, 63, 95544, 3709, 73, 95320, 23523, 65, 2925, 66]
    native = AutoTokenizer.from_pretrained(source, config=config, trust_remote_code=False)
    assert native.encode(text, add_special_tokens=False) == expected
    wrapped = Tokenicer.load(source, config=config, model_config=config, trust_remote_code=False)
    assert wrapped.encode(text, add_special_tokens=False) == expected
    messages = [{"role": "user", "content": text}]
    assert wrapped.apply_chat_template(messages, tokenize=False) == native.apply_chat_template(messages, tokenize=False)
    wrapped.save_pretrained(tmp_path)
    reloaded = Tokenicer.load(str(tmp_path), config=config, model_config=config, trust_remote_code=False)
    assert reloaded.encode(text, add_special_tokens=False) == expected


@pytest.mark.parametrize("override", [None, False])
@pytest.mark.parametrize("family", ["mistral", "laguna", "mixtral", "ministral3", "mistral_large4_text",
                                    "voxtral_realtime_text"])
def test_native_mistral_digit_regex_is_still_repaired(tmp_path, override, family):
    vocab = {token: i for i, token in enumerate(["<unk>", *sorted(pre_tokenizers.ByteLevel.alphabet()), "12", "123"])}
    for value in range(100_001 - len(vocab)):
        vocab[f"unused{value}"] = len(vocab)
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[("1", "2"), ("12", "3")], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(r"\p{N}{1,3}|\p{L}+|\s+|[^\p{L}\p{N}\s]+"), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token="<unk>")
    tokenizer.save_pretrained(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "mistral", "transformers_version": "4.36.0"}))
    metadata = json.loads((tmp_path / "tokenizer_config.json").read_text())
    metadata["tokenizer_class"] = "TokenizersBackend"
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(metadata))
    config = PreTrainedConfig(model_type=family, pad_token_id=0)
    kwargs = {} if override is None else {"fix_mistral_regex": override}
    wrapped = Tokenicer.load(str(tmp_path), config=config, model_config=config, trust_remote_code=False, **kwargs)
    expected = [vocab["1"], vocab["2"], vocab["3"]] if override is None else [vocab["123"]]
    assert backend.encode("123").ids == [vocab["123"]]
    assert wrapped.encode("123", add_special_tokens=False) == expected
