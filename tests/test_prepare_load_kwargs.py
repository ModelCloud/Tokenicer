# Copyright 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from tokenizers import AddedToken, ByteLevelBPETokenizer
from transformers import GPT2Config, GPT2Tokenizer

from tokenicer import Tokenicer

SPECIALS = [
    {"content": "<left>", "lstrip": True, "rstrip": False, "normalized": True, "single_word": False},
    {"content": "<right>", "lstrip": False, "rstrip": True, "normalized": False, "single_word": False},
]
TEXTS = ["Hello, world!", "中文测试 Hello 12345", "one   <left> two<right>   three", "def f():\n    return 123\n\tpass", "🙂 café\r\n"]


@pytest.fixture
def legacy_bundle(tmp_path):
    backend = ByteLevelBPETokenizer()
    backend.train_from_iterator(TEXTS, vocab_size=360, special_tokens=["<unk>", "<bos>", "<eos>", "<left>", "<right>"])
    backend.save_model(str(tmp_path))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "GPT2Tokenizer", "bos_token": "<bos>", "eos_token": "<eos>",
        "unk_token": "<unk>", "pad_token": "<unk>", "add_prefix_space": False,
    }))
    (tmp_path / "special_tokens_map.json").write_text(json.dumps({"additional_special_tokens": SPECIALS}))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "gpt2", "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0}))
    return tmp_path


def hashes(path):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in path.iterdir() if p.is_file()}


def reference(path):
    return GPT2Tokenizer.from_pretrained(str(path), trust_remote_code=False, additional_special_tokens=[
        AddedToken("<left>", lstrip=True, normalized=True, special=True),
        AddedToken("<right>", rstrip=True, normalized=False, special=True),
    ])


def test_native_tokenicer_preloads_legacy_metadata_and_roundtrips(legacy_bundle, tmp_path):
    path = legacy_bundle
    expected = reference(path)
    before = hashes(path)
    wrapped = Tokenicer.load(str(path), model_config=GPT2Config(pad_token_id=0), trust_remote_code=False)
    assert hashes(path) == before
    for text in TEXTS:
        assert wrapped.encode(text, add_special_tokens=False) == expected.encode(text, add_special_tokens=False)
    assert wrapped.added_tokens_decoder[3].lstrip
    assert wrapped.added_tokens_decoder[4].rstrip
    saved = tmp_path / "saved"
    wrapped.save_pretrained(saved)
    reloaded = Tokenicer.load(str(saved), model_config=GPT2Config(pad_token_id=0), trust_remote_code=False)
    for text in TEXTS:
        assert reloaded.encode(text, add_special_tokens=False) == expected.encode(text, add_special_tokens=False)


def test_preparer_does_not_construct_discover_or_normalize(legacy_bundle):
    config = SimpleNamespace(model_type="gpt2")
    original = {"config": config, "model_config": config, "trust_remote_code": False, "code_revision": "unexecuted"}
    before = hashes(legacy_bundle)
    with patch("transformers.AutoConfig.from_pretrained", side_effect=AssertionError("config discovery")), \
            patch("transformers.AutoTokenizer.from_pretrained", side_effect=AssertionError("constructor")), \
            patch.object(Tokenicer, "load", side_effect=AssertionError("instance normalization")):
        prepared = Tokenicer.prepare_load_kwargs(legacy_bundle, **original)
    assert prepared is not original
    assert all(prepared[k] is v for k, v in original.items())
    assert hashes(legacy_bundle) == before
    assert all(isinstance(t, AddedToken) and t.special for t in prepared["additional_special_tokens"])
    actual = GPT2Tokenizer.from_pretrained(str(legacy_bundle), **prepared)
    assert actual.encode(TEXTS[2]) == reference(legacy_bundle).encode(TEXTS[2])


@pytest.mark.parametrize("declared_class", [[], ["InternLM2Tokenizer"], {}, 42, None])
def test_preparer_leaves_nonstring_class_validation_to_selected_provider(legacy_bundle, declared_class):
    asset = legacy_bundle / "tokenizer_config.json"
    metadata = json.loads(asset.read_text())
    metadata["tokenizer_class"] = declared_class
    asset.write_text(json.dumps(metadata))
    prepared = Tokenicer.prepare_load_kwargs(legacy_bundle, config="caller-selected")
    assert prepared["config"] == "caller-selected"
    assert all(isinstance(token, AddedToken) for token in prepared["additional_special_tokens"])
    assert json.loads(asset.read_text())["tokenizer_class"] == declared_class


@pytest.mark.parametrize("value", [None, [], ["caller"], [AddedToken("caller", lstrip=True)]])
def test_explicit_values_are_authoritative(legacy_bundle, value):
    caller = {"additional_special_tokens": value, "bos_token": None, "padding_side": "left"}
    prepared = Tokenicer.prepare_load_kwargs(str(legacy_bundle), **caller)
    assert prepared["additional_special_tokens"] is value
    assert prepared["bos_token"] is None
    assert prepared["padding_side"] == "left"
    assert caller == {"additional_special_tokens": value, "bos_token": None, "padding_side": "left"}


def test_subfolder_ignores_decoy_metadata(legacy_bundle, tmp_path):
    selected = tmp_path / "selected"
    selected.mkdir()
    for path in list(legacy_bundle.iterdir()):
        if path.is_file():
            path.rename(selected / path.name)
    (tmp_path / "special_tokens_map.json").write_text("{bad decoy")
    prepared = Tokenicer.prepare_load_kwargs(tmp_path, subfolder="selected", local_files_only=True)
    actual = GPT2Tokenizer.from_pretrained(str(tmp_path), **prepared)
    assert actual.encode(TEXTS[2]) == reference(selected).encode(TEXTS[2])


def test_remote_selection_is_forwarded_without_model_discovery(legacy_bundle):
    selection = {"revision": "pinned-revision", "subfolder": "tokenizer", "cache_dir": "/selected/cache",
                 "token": "test-placeholder-not-a-secret", "local_files_only": True, "force_download": False,
                 "proxies": {"https": "http://test.invalid"}}
    def selected_file(repo, filename, **kwargs):
        assert repo == "owner/source"
        assert all(kwargs[k] == v for k, v in selection.items())
        assert not kwargs["_raise_exceptions_for_missing_entries"]
        assert "config" not in kwargs and "trust_remote_code" not in kwargs
        return str(legacy_bundle / filename)
    with patch("tokenicer.util.cached_file", side_effect=selected_file) as resolve:
        prepared = Tokenicer.prepare_load_kwargs("owner/source", **selection)
    assert resolve.call_count == 2
    assert all(prepared[k] is v for k, v in selection.items())
    assert prepared["additional_special_tokens"][0].lstrip


@pytest.mark.parametrize("broken", ["{broken", "[]", '{"additional_special_tokens":[{"content":42}]}'])
def test_bad_metadata_fails_then_same_path_repairs(legacy_bundle, broken):
    path = legacy_bundle / "special_tokens_map.json"
    before = path.read_bytes()
    path.write_text(broken)
    with pytest.raises((ValueError, TypeError)):
        Tokenicer.prepare_load_kwargs(legacy_bundle)
    path.write_bytes(before)
    prepared = Tokenicer.prepare_load_kwargs(legacy_bundle, trust_remote_code=False)
    assert GPT2Tokenizer.from_pretrained(str(legacy_bundle), **prepared).encode(TEXTS[2]) == reference(legacy_bundle).encode(TEXTS[2])


def test_native_constructor_failure_repair_has_no_stale_metadata(legacy_bundle):
    path = legacy_bundle / "vocab.json"
    before = path.read_bytes()
    path.write_text("{broken")
    with pytest.raises(Exception, match="Error while initializing BPE"):
        Tokenicer.load(str(legacy_bundle), model_config=GPT2Config(pad_token_id=0), trust_remote_code=False)
    path.write_bytes(before)
    wrapped = Tokenicer.load(str(legacy_bundle), model_config=GPT2Config(pad_token_id=0), trust_remote_code=False)
    assert wrapped.encode(TEXTS[2]) == reference(legacy_bundle).encode(TEXTS[2])


@pytest.mark.parametrize("modern", [False, True])
def test_config_addedtoken_flags_are_preserved_over_map(legacy_bundle, modern):
    config_path = legacy_bundle / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["bos_token"] = {"content": "<bos>", "single_word": True, "lstrip": False, "rstrip": False, "normalized": True}
    if modern:
        config["bos_token"]["__type"] = "AddedToken"
    config_path.write_text(json.dumps(config))
    map_path = legacy_bundle / "special_tokens_map.json"
    metadata = json.loads(map_path.read_text())
    metadata["bos_token"] = {"content": "<bos>", "single_word": False, "normalized": False}
    map_path.write_text(json.dumps(metadata))
    prepared = Tokenicer.prepare_load_kwargs(legacy_bundle)
    assert prepared["bos_token"].single_word
    assert prepared["bos_token"].normalized
    actual = GPT2Tokenizer.from_pretrained(str(legacy_bundle), **prepared)
    assert actual.added_tokens_decoder[1].single_word
    assert actual.added_tokens_decoder[1].normalized


def test_real_fallback_receives_same_normalized_metadata(legacy_bundle):
    with patch("tokenicer.tokenicer.AutoTokenizer.from_pretrained", side_effect=ValueError("injected native selection failure")) as first:
        wrapped = Tokenicer.load(str(legacy_bundle), model_config=GPT2Config(pad_token_id=0), trust_remote_code=False)
    assert first.call_count == 1
    assert all(isinstance(t, AddedToken) for t in first.call_args.kwargs["additional_special_tokens"])
    assert wrapped.encode(TEXTS[2]) == reference(legacy_bundle).encode(TEXTS[2])


@pytest.mark.parametrize("key, expected", [("bos_token", "<s>"), ("eos_token", "</s>"), ("unk_token", "<unk>")])
def test_llama_empty_token_compatibility_retains_explicit_none(tmp_path, key, expected):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"tokenizer_class": "LlamaTokenizer", key: ""}))
    assert Tokenicer.prepare_load_kwargs(tmp_path)[key] == expected
    assert Tokenicer.prepare_load_kwargs(tmp_path, **{key: None})[key] is None


MOSS_REPO = "fnlp/moss-moon-003-sft"
MOSS_REVISION = "3c06a7a241cef41caa9c339b149e9931d6b3deaa"
MOSS_HASHES = {
    "vocab.json": "53a5be5c089e133f66cfa2eb0889d91cfc76ab7ff9a3b0a237e6c550c50e8d40",
    "merges.txt": "b0686029221952e7145d99331c5d89dbf01ece4d86ddce52687a8dab23bcd048",
    "added_tokens.json": "1d60be8c005e6fcfb89a9aa675a5f6bb75f06638db35ce66f22ed9ce03127dae",
    "tokenizer_config.json": "26cb82be993f1738a94fb55643f24972a021553ec8453a84949c4bdad471edaf",
    "special_tokens_map.json": "e0eb0b3291bcd6004f59653eca3e393f235ab412c01b019efd70e15b62157058",
}
# Inspected source regex/byte/ranked-BPE oracle, independently calculated in
# GPT-QModel moss/tokenizer_oracle.json at commit 407aebac9fbcda9d9feef3595b2ab7dec5206550.
MOSS_CASES = [
    ("Hello, world!", [15496, 11, 995, 0]),
    ("中文测试 Hello 12345", [54119, 52926, 18435, 17031, 2231]),
    ("<|Human|>: 你好<eoh>\n<|MOSS|>: Hi<eom>", [27, 91, 20490, 91, 31175, 59163, 50331, 106067, 198, 27, 91, 44, 18420, 91, 31175, 15902, 106068]),
    ("def foo():\n    return 123\n\t\tpass", [4299, 22944, 33529, 198, 106056, 7783, 17031, 198, 106066, 6603]),
    ("We're I'M 9 1234567890  ", [1135, 821, 314, 6, 44, 860, 17031, 2231, 30924, 3829, 106058]),
    ("🙂 café\r\n", [8582, 25081, 40304, 64659]),
]


@pytest.fixture(scope="module")
def moss_assets():
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    paths = {name: Path(hf_hub_download(MOSS_REPO, name, revision=MOSS_REVISION)) for name in MOSS_HASHES}
    assert {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()} == MOSS_HASHES
    return paths["vocab.json"].parent


@pytest.mark.parametrize("trust", [False, True])
def test_pinned_moss_native_owned_consumer_and_serialization(moss_assets, tmp_path, trust):
    before = hashes(moss_assets)
    prepared = Tokenicer.prepare_load_kwargs(moss_assets, trust_remote_code=trust, local_files_only=True)
    with patch("transformers.AutoConfig.from_pretrained", side_effect=AssertionError("config discovery")), \
            patch("tokenicer.tokenicer.get_class_from_dynamic_module", side_effect=AssertionError("checkpoint import")):
        native = GPT2Tokenizer.from_pretrained(str(moss_assets), **prepared)
        config = GPT2Config(bos_token_id=106028, eos_token_id=106068, pad_token_id=106028)
        wrapped = Tokenicer.load(native, model_config=config, trust_remote_code=False)
    assert native.eos_token_id == wrapped.eos_token_id == 106068
    for text, ids in MOSS_CASES:
        assert native.encode(text, add_special_tokens=False) == ids
        assert wrapped.encode(text, add_special_tokens=False) == ids
    assert hashes(moss_assets) == before
    saved = tmp_path / "saved"
    wrapped.save_pretrained(saved)
    reloaded = Tokenicer.load(str(saved), config=config, model_config=config, trust_remote_code=False)
    for text, ids in MOSS_CASES:
        assert reloaded.encode(text, add_special_tokens=False) == ids
        assert reloaded.decode(ids, skip_special_tokens=False) == wrapped.decode(ids, skip_special_tokens=False)
        if "<eoh>" not in text and "<eom>" not in text:
            assert reloaded.decode(ids, skip_special_tokens=True) == text
    assert reloaded.eos_token_id == 106068


def test_pinned_moss_preparation_does_not_authorize_custom_class(moss_assets):
    with pytest.raises(ValueError, match="not available natively"):
        Tokenicer.load(str(moss_assets), model_config=GPT2Config(), trust_remote_code=False)
