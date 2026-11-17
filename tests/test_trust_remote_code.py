# Copyright 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from tokenicer import Tokenicer


def checkpoint(path, route):
    backend = Tokenizer(models.WordLevel(
        {"<unk>": 0, "<pad>": 1, "<s>": 2, "</s>": 3, "hello": 4, "world": 5},
        unk_token="<unk>",
    ))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>",
        bos_token="<s>", eos_token="</s>",
    )
    tokenizer.save_pretrained(path)
    metadata = json.loads((path / "tokenizer_config.json").read_text())
    module = "tokenization_poison" if route == "scan" else f"checkpoint_poison_{path.name}"
    marker = path / "EXECUTED"
    (path / f"{module}.py").write_text(
        "from pathlib import Path\n"
        "from transformers import PreTrainedTokenizerFast\n"
        f"Path({str(marker)!r}).write_text('checkpoint Python executed')\n"
        "class UnavailableCheckpointTokenizer(PreTrainedTokenizerFast):\n"
        "    pass\n"
    )
    metadata["tokenizer_class"] = "UnavailableCheckpointTokenizer"
    ref = f"{module}.UnavailableCheckpointTokenizer"
    if route == "dotted":
        metadata["tokenizer_class"] = ref
    elif route in {"auto_map", "auto_map_no_class"}:
        metadata["auto_map"] = {"AutoTokenizer": [ref, None]}
    elif route in {"legacy_auto_map", "legacy_auto_map_no_class"}:
        metadata["auto_map"] = [ref, None]
    if route.endswith("_no_class"):
        metadata.pop("tokenizer_class")
    (path / "tokenizer_config.json").write_text(json.dumps(metadata))
    # A valid tokenizer can outlive an unrecognized or broken model config.
    (path / "config.json").write_text(
        "{broken" if route in {"scan", "dotted"}
        else json.dumps({"model_type": "tokenicer_unregistered"})
    )
    return marker


@pytest.mark.parametrize("route", ["auto_map", "legacy_auto_map", "scan", "dotted", "unknown",
                                 "auto_map_no_class", "legacy_auto_map_no_class"])
@pytest.mark.parametrize("trust", [False, None, "omitted"])
def test_untrusted_custom_tokenizer_fails_closed_then_corrected_bundle_retries(tmp_path, monkeypatch, route, trust):
    marker = checkpoint(tmp_path, route)
    monkeypatch.syspath_prepend(str(tmp_path))
    kwargs = {} if trust == "omitted" else {"trust_remote_code": trust}
    with pytest.raises((ValueError, OSError)):
        try:
            Tokenicer.load(str(tmp_path), local_files_only=True, **kwargs)
        finally:
            assert not marker.exists(), "Untrusted checkpoint tokenizer Python executed"

    metadata = json.loads((tmp_path / "tokenizer_config.json").read_text())
    metadata["tokenizer_class"] = "PreTrainedTokenizerFast"
    metadata.pop("auto_map", None)
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(metadata))
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    tokenizer = Tokenicer.load(str(tmp_path), local_files_only=True, **kwargs)
    assert tokenizer.encode("hello world", add_special_tokens=False) == [4, 5]
    assert tokenizer.decode([4, 5]) == "hello world"
    assert tokenizer.pad_token_id == 1
    assert not marker.exists()


@pytest.mark.parametrize("route", ["auto_map", "legacy_auto_map", "scan", "dotted"])
def test_explicit_trust_retains_custom_tokenizer_loading(tmp_path, monkeypatch, route):
    marker = checkpoint(tmp_path, route)
    monkeypatch.syspath_prepend(str(tmp_path))
    tokenizer = Tokenicer.load(str(tmp_path), trust_remote_code=True, local_files_only=True)
    assert marker.read_text() == "checkpoint Python executed"
    assert type(tokenizer.tokenizer).__name__ == "UnavailableCheckpointTokenizer"
    assert tokenizer.encode("hello world", add_special_tokens=False) == [4, 5]


def test_supplied_tokenizer_instance_never_discovers_checkpoint_class(tmp_path):
    marker = checkpoint(tmp_path, "auto_map")
    native = PreTrainedTokenizerFast.from_pretrained(str(tmp_path), local_files_only=True)
    wrapped = Tokenicer.load(native, trust_remote_code=False)
    assert wrapped.tokenizer is native
    assert wrapped.encode("hello world", add_special_tokens=False) == [4, 5]
    assert not marker.exists()


def test_native_class_fallback_preserves_tokens_without_checkpoint_import(tmp_path):
    marker = checkpoint(tmp_path, "scan")
    metadata = json.loads((tmp_path / "tokenizer_config.json").read_text())
    metadata["tokenizer_class"] = "PreTrainedTokenizerFast"
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(metadata))
    tokenizer = Tokenicer.load(str(tmp_path), trust_remote_code=False, local_files_only=True)
    assert tokenizer.encode("hello world", add_special_tokens=False) == [4, 5]
    assert tokenizer.pad_token_id == 1
    assert not marker.exists()


@pytest.mark.parametrize("route", ["scan", "dotted"])
def test_class_resolution_itself_never_imports_untrusted_checkpoint(tmp_path, monkeypatch, route):
    marker = checkpoint(tmp_path, route)
    monkeypatch.syspath_prepend(str(tmp_path))
    metadata = json.loads((tmp_path / "tokenizer_config.json").read_text())
    resolved = Tokenicer._resolve_tokenizer_class(str(tmp_path), metadata["tokenizer_class"])
    assert not marker.exists()
    assert resolved is None
