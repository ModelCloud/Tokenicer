# Copyright 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from tokenicer import Tokenicer


def bundle(path):
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "hello": 3,
             "<|IMAGE_PLACEHOLDER|>": 4, "<|IMAGE_START|>": 5, "<|IMAGE_END|>": 6,
             "<|VIDEO_START|>": 7, "<|VIDEO_END|>": 8}
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
                                       unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<unk>")
    tokenizer.save_pretrained(path)
    values = json.loads((path / "tokenizer_config.json").read_text())
    marker = path / "EXECUTED"
    values.update(tokenizer_class="Ernie4_5_VLTokenizer", auto_map={"AutoTokenizer": ["hostile.Tokenizer", None]},
                  chat_template="<|image@placeholder|><|video@placeholder|>")
    (path / "tokenizer_config.json").write_text(json.dumps(values))
    (path / "hostile.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('hostile tokenizer executed')\n")
    (path / "config.json").write_text(json.dumps({"model_type": "ernie4_5_moe_vl", "auto_map": {"AutoConfig": "hostile.Config"}}))
    (path / "chat_template.json").write_text(json.dumps({"chat_template": "<|image@placeholder|><|video@placeholder|>"}))
    return marker


@pytest.mark.parametrize("trust", [False, True])
def test_native_metadata_loading_never_executes_checkpoint(tmp_path, trust):
    marker = bundle(tmp_path)
    wrapped = Tokenicer.load(str(tmp_path), model_config=SimpleNamespace(model_type="ernie4_5_moe_vl"), trust_remote_code=trust)
    assert not marker.exists()
    assert wrapped.image_token_id == 4
    assert wrapped.video_token_id == 9
    assert wrapped.tokenizer.encode("<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>") == [5, 4, 6]
    assert wrapped.chat_template == "<|IMAGE_PLACEHOLDER|><|VIDEO_PLACEHOLDER|>"


def test_explicit_overrides_and_save_reload(tmp_path):
    bundle(tmp_path)
    wrapped = Tokenicer.load(str(tmp_path), extra_special_tokens={"video_token": "custom-video"}, padding_side="left")
    assert wrapped.video_token == "custom-video"
    assert wrapped.padding_side == "left"
    saved = tmp_path / "saved"
    wrapped.save_pretrained(saved)
    reloaded = Tokenicer.load(str(saved), model_config=SimpleNamespace(model_type="ernie4_5_vl_moe"))
    assert reloaded.video_token == "custom-video"
    assert reloaded.image_token_id == 4
    assert reloaded.chat_template == wrapped.chat_template


def test_implicit_config_is_json_only_even_when_trusted(tmp_path):
    marker = bundle(tmp_path)
    tokenizer = Tokenicer.load(str(tmp_path), trust_remote_code=True)
    assert tokenizer.image_token_id == 4
    assert not marker.exists()


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("special_ids", [{}, {"bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0}])
def test_nested_json_config_materializes_text_metadata(tmp_path, trust, special_ids):
    marker = bundle(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "ernie4_5_moe_vl", "text_config": {"vocab_size": 103367, **special_ids},
        "auto_map": {"AutoConfig": "hostile.Config"},
    }))
    tokenizer = Tokenicer.load(str(tmp_path), trust_remote_code=trust)
    assert tokenizer.model_config.vocab_size == 103367
    assert tokenizer.pad_token_id == 0
    assert tokenizer.bos_token_id == 1
    assert tokenizer.eos_token_id == 2
    assert not marker.exists()


@pytest.mark.parametrize("trust", [False, True])
@pytest.mark.parametrize("has_pad", [False, True])
def test_tokenizer_only_bundle_preserves_pad_fallback(tmp_path, trust, has_pad):
    marker = bundle(tmp_path)
    (tmp_path / "config.json").unlink()
    if not has_pad:
        metadata = json.loads((tmp_path / "tokenizer_config.json").read_text())
        metadata.pop("pad_token", None)
        (tmp_path / "tokenizer_config.json").write_text(json.dumps(metadata))
    tokenizer = Tokenicer.load(str(tmp_path), trust_remote_code=trust)
    assert tokenizer.model_config is None
    assert tokenizer.pad_token_id == (0 if has_pad else 2)
    assert tokenizer.image_token_id == 4
    assert not marker.exists()


def test_malformed_config_errors_then_corrected_nested_config_retries(tmp_path):
    marker = bundle(tmp_path)
    (tmp_path / "config.json").write_text("{broken")
    with pytest.raises(OSError):
        Tokenicer.load(str(tmp_path), trust_remote_code=True)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "ernie4_5_moe_vl", "text_config": {"vocab_size": 16}}))
    tokenizer = Tokenicer.load(str(tmp_path), trust_remote_code=True)
    assert tokenizer.model_config.vocab_size == 16
    assert not marker.exists()


def test_invalid_native_assets_fail_closed_then_retry(tmp_path, monkeypatch):
    from transformers import AutoTokenizer

    marker = bundle(tmp_path)
    serialized = (tmp_path / "tokenizer.json").read_bytes()
    (tmp_path / "tokenizer.json").unlink()
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: pytest.fail("Auto dispatch was reached"))
    with pytest.raises((ValueError, OSError)):
        Tokenicer.load(str(tmp_path), trust_remote_code=True)
    assert not marker.exists()
    (tmp_path / "tokenizer.json").write_bytes(serialized)
    assert Tokenicer.load(str(tmp_path), trust_remote_code=True).image_token_id == 4
    assert not marker.exists()


@pytest.mark.parametrize("trust", [False, True])
def test_pinned_public_sentencepiece_bundle(tmp_path, trust):
    import shutil

    from huggingface_hub import snapshot_download

    revision = "e3815e65c607ea211bfe21b46ab0cd264b76731c"
    files = ["tokenizer.model", "tokenizer_config.json", "added_tokens.json", "special_tokens_map.json", "chat_template.json"]
    source = snapshot_download("baidu/ERNIE-4.5-VL-28B-A3B-PT", revision=revision, allow_patterns=files)
    from pathlib import Path
    for name in files:
        shutil.copyfile(Path(source) / name, tmp_path / name)
    values = json.loads((tmp_path / "tokenizer_config.json").read_text())
    values["auto_map"] = {"AutoTokenizer": ["hostile.Tokenizer", None]}
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(values))
    marker = tmp_path / "EXECUTED"
    (tmp_path / "hostile.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('hostile tokenizer')\n")
    tokenizer = Tokenicer.load(str(tmp_path), model_config=SimpleNamespace(model_type="ernie4_5_moe_vl"),
                               trust_remote_code=trust)
    assert not marker.exists()
    assert tokenizer.encode("Hello world!  中文\nX") == [16276, 3135, 94018, 269, 11422, 23, 94026]
    assert tokenizer.image_token_id == 100295
    assert tokenizer.video_token_id == 103367
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": "Describe."}]}], tokenize=False)
    assert "Picture 1:<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>Describe." in prompt
    assert tokenizer.encode("<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>") == [101304, 100295, 101305]
