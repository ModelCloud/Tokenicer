# Copyright 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, LlamaConfig, LlamaTokenizer, Phi3Config

from tokenicer import Tokenicer
from tokenicer.sentencepiece import normalize_sentencepiece

REFERENCE = json.loads((Path(__file__).parent / "fixtures/sentencepiece_reference.json").read_text())
FILES = ("tokenizer.model", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json")


@pytest.fixture(params=REFERENCE, scope="module")
def source(request, tmp_path_factory):
    reference = REFERENCE[request.param]
    path = tmp_path_factory.mktemp(request.param)
    for name in FILES:
        asset = hf_hub_download(reference["repo"], name, revision=reference["revision"])
        shutil.copyfile(asset, path / name)
    assert {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in FILES} == reference["asset_sha256"]
    return path, reference


def config(reference):
    return Phi3Config(pad_token_id=0) if reference["metadata"]["add_bos_token"] else LlamaConfig(pad_token_id=0)


def assert_source(tokenizer, reference):
    for text, ids in reference["source_fast_ids"].items():
        assert tokenizer.encode(text) == ids, repr(text)
    for token_id, token in reference["metadata"]["added_tokens_decoder"].items():
        assert tokenizer.convert_tokens_to_ids(token["content"]) == int(token_id)
        actual = tokenizer.added_tokens_decoder[int(token_id)]
        assert actual.content == token["content"]
        for flag in ("lstrip", "rstrip", "normalized", "single_word"):
            assert getattr(actual, flag) == token[flag]


def pipeline(tokenizer):
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    return {key: backend[key] for key in ("normalizer", "pre_tokenizer", "decoder", "model")}


@pytest.mark.parametrize("trust", [None, False, True])
@pytest.mark.parametrize("loaded_instance", [False, True])
def test_real_native_source_fast_ids_and_authoritative_added_ids(source, trust, loaded_instance):
    path, reference = source
    before = {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in FILES}
    value = LlamaTokenizer.from_pretrained(path, trust_remote_code=False) if loaded_instance else str(path)
    with patch("transformers.AutoConfig.from_pretrained", side_effect=AssertionError("config discovery")), \
            patch("tokenicer.tokenicer.get_class_from_dynamic_module", side_effect=AssertionError("checkpoint code")), \
            patch("tokenicer.tokenicer.normalize_native_sentencepiece", wraps=normalize_sentencepiece) as normalize, \
            patch("tokenicer.tokenicer.AutoTokenizer.from_pretrained", wraps=AutoTokenizer.from_pretrained) as construct:
        wrapped = Tokenicer.load(
            value, model_config=config(reference), config=LlamaConfig(),
            trust_remote_code=trust, normalize_sentencepiece=True,
        )
    normalize.assert_called_once()
    assert construct.call_count == int(not loaded_instance)
    if not loaded_instance:
        assert "normalize_sentencepiece" not in construct.call_args.kwargs
    assert type(wrapped.tokenizer) is LlamaTokenizer
    assert_source(wrapped, reference)
    assert before == {name: hashlib.sha256((path / name).read_bytes()).hexdigest() for name in FILES}


def test_native_instance_and_path_two_serialized_roundtrips(source, tmp_path):
    path, reference = source
    wrapped = Tokenicer.load(LlamaTokenizer.from_pretrained(path), model_config=config(reference), normalize_sentencepiece=True)
    assert_source(wrapped, reference)
    for cycle in range(2):
        saved = tmp_path / str(cycle)
        wrapped.save_pretrained(saved)
        assert json.loads((saved / "tokenizer_config.json").read_text())["tokenizer_class"] == "LlamaTokenizer"
        backend = json.loads((saved / "tokenizer.json").read_text())
        assert backend["normalizer"] is not None
        assert not (saved / "tokenizer.model").exists()
        native = LlamaTokenizer.from_pretrained(saved)
        instance = Tokenicer.load(native, model_config=config(reference))
        wrapped = Tokenicer.load(str(saved), model_config=config(reference), config=LlamaConfig())
        assert type(instance.tokenizer) is type(wrapped.tokenizer) is LlamaTokenizer
        assert_source(instance, reference)
        assert_source(wrapped, reference)
        assert pipeline(wrapped) == pipeline(instance)


@pytest.mark.parametrize("key", ["legacy", "add_prefix_space"])
@pytest.mark.parametrize("value", [None, False, True])
def test_explicit_native_caller_flags_remain_authoritative(source, key, value):
    path, reference = source
    expected = LlamaTokenizer.from_pretrained(path, **{key: value})
    wrapped = Tokenicer.load(str(path), model_config=config(reference), config=LlamaConfig(), normalize_sentencepiece=True, **{key: value})
    assert pipeline(wrapped) == pipeline(expected)
    for text in reference["source_fast_ids"]:
        assert wrapped.encode(text) == expected.encode(text)


def test_native_opt_out_and_owned_provider_are_not_changed(source):
    path, reference = source
    native = LlamaTokenizer.from_pretrained(path)
    before = pipeline(native)
    wrapped = Tokenicer.load(native, model_config=config(reference), normalize_sentencepiece=False)
    assert pipeline(wrapped) == before
    wrapped = Tokenicer.load(LlamaTokenizer.from_pretrained(path), model_config=config(reference))
    assert pipeline(wrapped) == before

    class OwnedTokenizer(LlamaTokenizer):
        pass

    owned = OwnedTokenizer.from_pretrained(path)
    before = pipeline(owned)
    wrapped = Tokenicer.load(owned, model_config=config(reference), normalize_sentencepiece=True)
    assert type(wrapped.tokenizer) is OwnedTokenizer
    assert pipeline(wrapped) == before


def test_selected_subfolder_ignores_root_decoy_and_roundtrips(source, tmp_path):
    path, reference = source
    selected = tmp_path / "selected"
    shutil.copytree(path, selected)
    (tmp_path / "tokenizer.json").write_text("invalid decoy")
    (tmp_path / "tokenizer_config.json").write_text("invalid decoy")
    wrapped = Tokenicer.load(str(tmp_path), subfolder="selected", model_config=config(reference), config=LlamaConfig(), normalize_sentencepiece=True)
    assert_source(wrapped, reference)
    saved = selected / "saved"
    wrapped.save_pretrained(saved)
    wrapped = Tokenicer.load(str(selected), subfolder="saved", model_config=config(reference), config=LlamaConfig())
    assert_source(wrapped, reference)


def test_failed_assets_then_same_path_restored_retry(source, tmp_path):
    path, reference = source
    selected = tmp_path / "retry"
    shutil.copytree(path, selected)
    asset = selected / "tokenizer.model"
    original = asset.read_bytes()
    asset.write_bytes(b"not a sentencepiece model")
    with pytest.raises(Exception):
        Tokenicer.load(str(selected), model_config=config(reference), config=LlamaConfig(), normalize_sentencepiece=True)
    asset.write_bytes(original)
    wrapped = Tokenicer.load(str(selected), model_config=config(reference), config=LlamaConfig(), normalize_sentencepiece=True)
    assert_source(wrapped, reference)
    saved = tmp_path / "saved"
    wrapped.save_pretrained(saved)
    asset = saved / "tokenizer.json"
    original = asset.read_bytes()
    asset.write_text("invalid tokenizer JSON")
    with pytest.raises(Exception):
        Tokenicer.load(str(saved), model_config=config(reference), config=LlamaConfig())
    asset.write_bytes(original)
    wrapped = Tokenicer.load(str(saved), model_config=config(reference), config=LlamaConfig())
    assert_source(wrapped, reference)


def test_serialized_pipeline_opt_out_remains_native(source, tmp_path):
    path, reference = source
    wrapped = Tokenicer.load(LlamaTokenizer.from_pretrained(path), model_config=config(reference), normalize_sentencepiece=True)
    wrapped.save_pretrained(tmp_path)
    native = LlamaTokenizer.from_pretrained(tmp_path)
    wrapped = Tokenicer.load(
        str(tmp_path), model_config=config(reference), config=LlamaConfig(),
        normalize_sentencepiece=True, prefer_serialized_tokenizer=False,
    )
    assert pipeline(wrapped) == pipeline(native)


def test_loaded_remote_instance_does_not_guess_another_revision(source, tmp_path):
    path, reference = source
    wrapped = Tokenicer.load(LlamaTokenizer.from_pretrained(path), model_config=config(reference), normalize_sentencepiece=True)
    wrapped.save_pretrained(tmp_path)
    native = LlamaTokenizer.from_pretrained(tmp_path)
    native.name_or_path = "owner/unselected-revision"
    before = pipeline(native)
    with patch("tokenicer.util.cached_file", side_effect=AssertionError("unselected Hub revision")):
        wrapped = Tokenicer.load(native, model_config=config(reference), normalize_sentencepiece=True)
    assert pipeline(wrapped) == before
    selection = {"revision": reference["revision"], "subfolder": "selected", "local_files_only": True}

    def selected_file(repo, filename, **kwargs):
        assert repo == "owner/unselected-revision"
        assert filename == "tokenizer.json"
        assert all(kwargs[key] == value for key, value in selection.items())
        return str(tmp_path / filename)

    with patch("tokenicer.util.cached_file", side_effect=selected_file) as resolve:
        wrapped = Tokenicer.load(native, model_config=config(reference), **selection)
    resolve.assert_called_once()
    assert_source(wrapped, reference)


def test_unsupported_sentencepiece_normalizer_is_not_changed(source, tmp_path):
    from sentencepiece import sentencepiece_model_pb2

    path, reference = source
    shutil.copytree(path, tmp_path, dirs_exist_ok=True)
    asset = tmp_path / "tokenizer.model"
    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString(asset.read_bytes())
    proto.normalizer_spec.remove_extra_whitespaces = True
    asset.write_bytes(proto.SerializeToString())
    native = LlamaTokenizer.from_pretrained(tmp_path)
    before = pipeline(native)
    wrapped = Tokenicer.load(native, model_config=config(reference), normalize_sentencepiece=True)
    assert pipeline(wrapped) == before
