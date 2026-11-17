# Copyright 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import pytest
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import LlamaConfig, TokenizersBackend

from tokenicer import Tokenicer


REFERENCE = json.loads((Path(__file__).parent / "fixtures/legacy_metaspace_reference.json").read_text())


def bundle(path, prefix=False, nested=True):
    path.mkdir(exist_ok=True)
    backend = Tokenizer(models.WordLevel({"<unk>": 0, "▁Hello": 1, "Hello": 2, "▁world": 3}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Metaspace(prepend_scheme="always" if prefix else "never")
    backend.decoder = decoders.Metaspace(prepend_scheme="always" if prefix else "never")
    spec = json.loads(backend.to_str())
    for field, children in (("pre_tokenizer", "pretokenizers"), ("decoder", "decoders")):
        node = spec[field]
        node.pop("prepend_scheme")
        node["add_prefix_space"] = prefix
        if nested:
            node = {"type": "Sequence", children: [{"type": "Sequence", children: [node]}]}
        spec[field] = node
    (path / "tokenizer.json").write_text(json.dumps(spec))
    (path / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<unk>", "pad_token": "<unk>",
    }))
    return spec


def hashes(path):
    return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in path.iterdir() if f.is_file()}


def assert_oracle(tokenizer, reference):
    for text, ids in reference["ids"].items():
        assert tokenizer.encode(text, add_special_tokens=False) == ids, repr(text)
        assert tokenizer.decode(ids, skip_special_tokens=False) == reference["decode"][text], repr(text)


@pytest.fixture(params=REFERENCE, scope="module")
def source(request, tmp_path_factory):
    ref = REFERENCE[request.param]
    path = tmp_path_factory.mktemp(request.param)
    for name, digest in ref["asset_sha256"].items():
        src = hf_hub_download(ref["repo"], name, revision=ref["revision"])
        shutil.copyfile(src, path / name)
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == digest
    return path, ref


@pytest.mark.parametrize("owned", [False, True])
def test_real_legacy_oracle_path_and_owned_constructor_two_cycles(source, tmp_path, owned):
    path, ref = source
    before = hashes(path)
    options = {"padding_side": "left", "use_fast": True, "cache_dir": str(tmp_path / "cache")}
    config = LlamaConfig(pad_token_id=0)
    with patch("tokenicer.tokenicer.get_class_from_dynamic_module", side_effect=AssertionError("checkpoint code")):
        if owned:
            prepared = Tokenicer.prepare_load_kwargs(path, **options)
            assert all(prepared[k] == v for k, v in options.items())
            value = TokenizersBackend.from_pretrained(path, **prepared)
        else:
            value = str(path)
        tokenizer = Tokenicer.load(value, model_config=config, config=config, trust_remote_code=False, **options)
    assert_oracle(tokenizer, ref)
    assert tokenizer.padding_side == "left"
    assert hashes(path) == before
    for cycle in range(2):
        saved = tmp_path / f"saved-{cycle}"
        tokenizer.save_pretrained(saved)
        tokenizer = Tokenicer.load(str(saved), model_config=config, config=config, trust_remote_code=False)
        assert_oracle(tokenizer, ref)


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("prefix", [False, True])
def test_legacy_true_false_and_nested_serializer_fields(tmp_path, prefix, nested):
    path = tmp_path / "source"
    spec = bundle(path, prefix, nested)
    before = hashes(path)
    kwargs = Tokenicer.prepare_load_kwargs(path, cache_dir=tmp_path / "cache")
    expected = json.loads(json.dumps(spec))
    for field, children in (("pre_tokenizer", "pretokenizers"), ("decoder", "decoders")):
        node = expected[field][children][0][children][0] if nested else expected[field]
        del node["add_prefix_space"]
        node["prepend_scheme"] = "always" if prefix else "never"
    converted = Path(kwargs["tokenizer_file"])
    assert json.loads(converted.read_bytes()) == expected
    assert converted.stem == hashlib.sha256(converted.read_bytes()).hexdigest()
    tokenizer = TokenizersBackend.from_pretrained(path, **kwargs)
    assert tokenizer.encode("Hello world", add_special_tokens=False) == [1 if prefix else 2, 3]
    assert tokenizer.decode([1 if prefix else 2, 3]) == "Hello world"
    assert tokenizer.encode(" Hello world", add_special_tokens=False) == [1, 3]
    assert tokenizer.decode([1, 3]) == ("Hello world" if prefix else " Hello world")
    assert hashes(path) == before


@pytest.mark.parametrize("field", ["pre_tokenizer", "decoder"])
def test_each_pipeline_is_migrated_independently(tmp_path, field):
    spec = bundle(tmp_path, nested=False)
    other = "decoder" if field == "pre_tokenizer" else "pre_tokenizer"
    spec[other].pop("add_prefix_space")
    spec[other]["prepend_scheme"] = "never"
    (tmp_path / "tokenizer.json").write_text(json.dumps(spec))
    prepared = Tokenicer.prepare_load_kwargs(tmp_path, cache_dir=tmp_path / "cache")
    tokenizer = TokenizersBackend.from_pretrained(tmp_path, **prepared)
    assert tokenizer.encode("Hello world", add_special_tokens=False) == [2, 3]
    assert tokenizer.decode([2, 3]) == "Hello world"


@pytest.mark.parametrize("scheme", [None, "always", "never", "first"])
def test_absent_legacy_and_explicit_modern_schemes_do_not_write_cache(tmp_path, scheme):
    spec = bundle(tmp_path, nested=False)
    for field in ("pre_tokenizer", "decoder"):
        spec[field].pop("add_prefix_space")
        if scheme is not None:
            spec[field]["prepend_scheme"] = scheme
    (tmp_path / "tokenizer.json").write_text(json.dumps(spec))
    before = hashes(tmp_path)
    prepared = Tokenicer.prepare_load_kwargs(tmp_path, cache_dir=tmp_path / "cache")
    assert "tokenizer_file" not in prepared
    assert not (tmp_path / "cache").exists()
    tokenizer = TokenizersBackend.from_pretrained(tmp_path, **prepared)
    assert tokenizer.encode("Hello world", add_special_tokens=False) == [2 if scheme == "never" else 1, 3]
    assert hashes(tmp_path) == before


@pytest.mark.parametrize("key,value", [("tokenizer_file", None), ("tokenizer_file", "explicit.json"),
                                       ("tokenizer_object", None), ("tokenizer_object", object()), ("gguf_file", "model.gguf")])
def test_explicit_file_object_none_and_gguf_are_authoritative(tmp_path, key, value):
    bundle(tmp_path)
    with patch("tokenicer.serialized.serialized_tokenizer_file", side_effect=AssertionError("default file selected")):
        prepared = Tokenicer.prepare_load_kwargs(tmp_path, **{key: value})
    assert prepared[key] is value
    if key != "tokenizer_file":
        assert "tokenizer_file" not in prepared


@pytest.mark.parametrize("legacy,scheme", [(False, "never"), (True, "always"), (True, "first"), (False, "always")])
def test_modern_scheme_is_not_overridden_or_conflict_silenced(tmp_path, legacy, scheme):
    spec = bundle(tmp_path, prefix=legacy, nested=False)
    for field in ("pre_tokenizer", "decoder"):
        spec[field]["prepend_scheme"] = scheme
    (tmp_path / "tokenizer.json").write_text(json.dumps(spec))
    prepared = Tokenicer.prepare_load_kwargs(tmp_path, cache_dir=tmp_path / "cache")
    assert "tokenizer_file" not in prepared
    if not legacy and scheme == "always":
        with pytest.raises(Exception, match="add_prefix_space does not match"):
            TokenizersBackend.from_pretrained(tmp_path, **prepared)
    else:
        tokenizer = TokenizersBackend.from_pretrained(tmp_path, **prepared)
        assert tokenizer.encode("Hello world", add_special_tokens=False) == [1 if legacy else 2, 3]


@pytest.mark.parametrize("broken", [None, 0, "false"])
def test_malformed_legacy_flag_fails_before_cache_then_repairs(tmp_path, broken):
    spec = bundle(tmp_path, nested=False)
    source = tmp_path / "tokenizer.json"
    original = source.read_bytes()
    spec["decoder"]["add_prefix_space"] = broken
    source.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="Metaspace add_prefix_space must be a boolean"):
        Tokenicer.prepare_load_kwargs(tmp_path, cache_dir=tmp_path / "cache")
    assert not (tmp_path / "cache").exists()
    source.write_bytes(original)
    prepared = Tokenicer.prepare_load_kwargs(tmp_path, cache_dir=tmp_path / "cache")
    assert TokenizersBackend.from_pretrained(tmp_path, **prepared).encode("Hello world", add_special_tokens=False) == [2, 3]


def test_selected_subfolder_and_versioned_file_ignore_root_decoys(tmp_path):
    bundle(tmp_path, prefix=True)
    selected = tmp_path / "selected"
    bundle(selected)
    (selected / "tokenizer.json").rename(selected / "tokenizer.0.0.1.json")
    metadata = json.loads((selected / "tokenizer_config.json").read_text())
    metadata["fast_tokenizer_files"] = ["tokenizer.0.0.1.json"]
    (selected / "tokenizer_config.json").write_text(json.dumps(metadata))
    (selected / "tokenizer.json").write_text("{invalid decoy")
    prepared = Tokenicer.prepare_load_kwargs(tmp_path, subfolder="selected", cache_dir=tmp_path / "cache")
    tokenizer = TokenizersBackend.from_pretrained(tmp_path, **prepared)
    assert tokenizer.encode("Hello world", add_special_tokens=False) == [2, 3]


def test_remote_file_selection_and_caller_options_are_preserved(tmp_path):
    bundle(tmp_path)
    selection = {"revision": "fixed-revision", "subfolder": "chosen", "token": "placeholder-not-a-secret",
                 "cache_dir": tmp_path / "cache", "local_files_only": True, "force_download": False,
                 "proxies": {"https": "http://invalid.test"}, "repo_type": "model"}
    options = {"padding_side": "left", "use_fast": True, "add_prefix_space": False, "trust_remote_code": True}
    def resolve(repo, filename, **kwargs):
        assert repo == "owner/tokenizer"
        assert kwargs == {**selection, "_raise_exceptions_for_missing_entries": False}
        p = tmp_path / filename
        return str(p) if p.is_file() else None
    with patch("tokenicer.util.cached_file", side_effect=resolve):
        prepared = Tokenicer.prepare_load_kwargs("owner/tokenizer", **selection, **options)
    assert all(prepared[k] is v for k, v in {**selection, **options}.items())
    tokenizer = TokenizersBackend(tokenizer_file=prepared["tokenizer_file"])
    assert tokenizer.encode("Hello world", add_special_tokens=False) == [2, 3]


def test_content_addressed_cache_reuse_repair_and_source_change(tmp_path):
    path = tmp_path / "source"
    spec = bundle(path)
    options = {"cache_dir": tmp_path / "cache"}
    first = Tokenicer.prepare_load_kwargs(path, **options)
    cached = Path(first["tokenizer_file"])
    before = cached.read_bytes()
    stat = cached.stat().st_mtime_ns
    assert Tokenicer.prepare_load_kwargs(path, **options) == first
    assert cached.stat().st_mtime_ns == stat
    cached.write_text("{corrupt")
    assert Tokenicer.prepare_load_kwargs(path, **options) == first
    assert cached.read_bytes() == before
    source = path / "tokenizer.json"
    original = source.read_bytes()
    source.write_text("{corrupt source")
    with pytest.raises(ValueError):
        Tokenicer.prepare_load_kwargs(path, **options)
    source.write_bytes(original)
    assert Tokenicer.prepare_load_kwargs(path, **options) == first
    for field, child in (("pre_tokenizer", "pretokenizers"), ("decoder", "decoders")):
        spec[field][child][0][child][0]["add_prefix_space"] = True
    source.write_text(json.dumps(spec))
    second = Tokenicer.prepare_load_kwargs(path, **options)
    assert second["tokenizer_file"] != first["tokenizer_file"]
    assert TokenizersBackend.from_pretrained(path, **first).encode("Hello world", add_special_tokens=False) == [2, 3]
    assert TokenizersBackend.from_pretrained(path, **second).encode("Hello world", add_special_tokens=False) == [1, 3]


def test_atomic_publication_failure_cleans_partial_then_retry_works(tmp_path):
    bundle(tmp_path)
    with patch("tokenicer.serialized.os.replace", side_effect=OSError("injected publish failure")) as publish:
        with pytest.raises(OSError, match="injected publish failure"):
            Tokenicer.prepare_load_kwargs(tmp_path, cache_dir=tmp_path / "cache")
    publish.assert_called_once()
    assert list((tmp_path / "cache").rglob("*json")) == []
    assert not list((tmp_path / "cache" / "tokenicer" / "metaspace-v1").iterdir())
    prepared = Tokenicer.prepare_load_kwargs(tmp_path, cache_dir=tmp_path / "cache")
    assert TokenizersBackend.from_pretrained(tmp_path, **prepared).encode("Hello world", add_special_tokens=False) == [2, 3]


def test_concurrent_cold_cache_publishes_one_complete_immutable_file(tmp_path):
    bundle(tmp_path)
    barrier = threading.Barrier(4)
    def prepare(_):
        barrier.wait(timeout=10)
        return Tokenicer.prepare_load_kwargs(tmp_path, cache_dir=tmp_path / "cache")["tokenizer_file"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(prepare, range(4), timeout=20))
    assert len(set(paths)) == 1
    assert list(Path(paths[0]).parent.iterdir()) == [Path(paths[0])]
    assert Tokenizer.from_file(paths[0]).encode("Hello world").ids == [2, 3]


def test_native_fallback_reuses_same_prepared_file_without_constructor_config(tmp_path):
    bundle(tmp_path)
    config = LlamaConfig(pad_token_id=0)
    with patch("tokenicer.tokenicer.AutoTokenizer.from_pretrained", side_effect=ValueError("native selection failed")) as primary, \
            patch.object(TokenizersBackend, "from_pretrained", wraps=TokenizersBackend.from_pretrained) as fallback:
        tokenizer = Tokenicer.load(str(tmp_path), model_config=config, config=config, cache_dir=tmp_path / "cache")
    assert primary.call_args.kwargs["config"] is config
    assert primary.call_args.kwargs["tokenizer_file"] == fallback.call_args.kwargs["tokenizer_file"]
    assert "config" not in fallback.call_args.kwargs
    assert tokenizer.encode("Hello world", add_special_tokens=False) == [2, 3]
    tokenizer.save_pretrained(tmp_path / "saved")


def test_read_only_source_uses_separate_writable_cache_without_constructor(tmp_path):
    path = tmp_path / "source"
    bundle(path)
    before = hashes(path)
    path.chmod(0o555)
    for child in path.iterdir():
        child.chmod(0o444)
    try:
        with patch("transformers.AutoTokenizer.from_pretrained", side_effect=AssertionError("constructor")), \
                patch("transformers.AutoConfig.from_pretrained", side_effect=AssertionError("config discovery")), \
                patch.object(Tokenicer, "load", side_effect=AssertionError("instance normalization")):
            prepared = Tokenicer.prepare_load_kwargs(path, cache_dir=tmp_path / "cache")
        assert hashes(path) == before
        assert TokenizersBackend.from_pretrained(path, **prepared).encode("Hello world", add_special_tokens=False) == [2, 3]
    finally:
        path.chmod(0o755)
        for child in path.iterdir():
            child.chmod(0o644)


@pytest.mark.parametrize("as_object", [False, True])
def test_explicit_real_file_or_object_wins_over_broken_default_json(tmp_path, as_object):
    bundle(tmp_path)
    backend = Tokenizer(models.WordLevel({"<unk>": 0, "Hello": 1, "world": 2}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    filename = tmp_path / "explicit.json"
    backend.save(str(filename))
    options = {"tokenizer_object": backend} if as_object else {"tokenizer_file": str(filename)}
    config = LlamaConfig(pad_token_id=0)
    tokenizer = Tokenicer.load(str(tmp_path), model_config=config, config=config, trust_remote_code=False, **options)
    assert tokenizer.encode("Hello world", add_special_tokens=False) == [1, 2]


def test_native_retry_selects_subfolder_class_not_root_decoy(tmp_path):
    bundle(tmp_path, prefix=True)
    root = tmp_path / "tokenizer_config.json"
    metadata = json.loads(root.read_text())
    metadata["tokenizer_class"] = "GPT2Tokenizer"
    root.write_text(json.dumps(metadata))
    selected = tmp_path / "selected"
    bundle(selected)
    config = LlamaConfig(pad_token_id=0)
    with patch("tokenicer.tokenicer.AutoTokenizer.from_pretrained", side_effect=ValueError("force selected retry")):
        tokenizer = Tokenicer.load(str(tmp_path), model_config=config, config=config, subfolder="selected", cache_dir=tmp_path / "cache")
    assert type(tokenizer.tokenizer) is TokenizersBackend
    assert tokenizer.encode("Hello world", add_special_tokens=False) == [2, 3]
