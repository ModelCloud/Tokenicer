# Copyright 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest
from huggingface_hub import hf_hub_download
from test_serialized_tokenizer import PROBES, _build_byte_level_tokenizer, _write_checkpoint
from transformers import LlamaConfig, TokenizersBackend

from tokenicer import Tokenicer


@pytest.mark.parametrize("fail_primary", [False, True])
def test_selected_model_config_is_constructor_only_and_two_saves_work(tmp_path, fail_primary):
    backend = _build_byte_level_tokenizer()
    _write_checkpoint(str(tmp_path), backend, "LlamaTokenizerFast")
    config = LlamaConfig(pad_token_id=0)
    before = config.to_dict()
    def load():
        return Tokenicer.load(str(tmp_path), model_config=config, config=config, trust_remote_code=False)
    if fail_primary:
        with patch("tokenicer.tokenicer.AutoTokenizer.from_pretrained", side_effect=ValueError("forced fallback")):
            tokenizer = load()
    else:
        tokenizer = load()
    assert type(tokenizer.tokenizer) is TokenizersBackend
    assert "config" not in tokenizer.tokenizer.init_kwargs
    for cycle in range(2):
        for text in PROBES:
            assert tokenizer.encode(text, add_special_tokens=False) == backend.encode(text, add_special_tokens=False).ids
        saved = tmp_path / f"saved-{cycle}"
        tokenizer.save_pretrained(saved)
        assert "config" not in json.loads((saved / "tokenizer_config.json").read_text())
        tokenizer = Tokenicer.load(str(saved), model_config=config, config=config, trust_remote_code=False)
    assert config.to_dict() == before


def test_real_unlimited_ocr_hostile_metadata_serialized_fallback_two_cycles(tmp_path):
    repo = "baidu/Unlimited-OCR"
    revision = "8249d20b2822a7d8a4c22c8d8c9069b5ac36bda3"
    digests = {
        "tokenizer.json": "a02f8fd5228c90256bb4f6554c34a579d48f909e5beb232dc4afad870b55a8b4",
        "tokenizer_config.json": "a0cbe8464049da1f891b7a12676de06af4cb54c130995d42f71adc1c30c6e9f3",
        "special_tokens_map.json": "ab4bd57ce17d62e39e0a39e739de1e407484f090f0b2c7e391312bca7a5b061a",
    }
    for filename, digest in digests.items():
        source = hf_hub_download(repo, filename, revision=revision)
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == digest
        shutil.copyfile(source, tmp_path / filename)
    metadata = json.loads((tmp_path / "tokenizer_config.json").read_text())
    metadata["auto_map"] = {"AutoTokenizer": ["evil.Tokenizer", None]}
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(metadata))
    config = LlamaConfig(vocab_size=129280)
    with patch("tokenicer.tokenicer.get_class_from_dynamic_module", side_effect=AssertionError("checkpoint code")):
        tokenizer = Tokenicer.load(str(tmp_path), model_config=config, config=config, trust_remote_code=False)
    assert type(tokenizer.tokenizer) is TokenizersBackend
    for cycle in range(2):
        assert tokenizer.encode("Hello café 你好!", add_special_tokens=False) == [19923, 57664, 223, 30594, 3]
        assert "config" not in tokenizer.tokenizer.init_kwargs
        saved = tmp_path / f"saved-{cycle}"
        tokenizer.save_pretrained(saved)
        tokenizer = Tokenicer.load(str(saved), model_config=config, config=config, trust_remote_code=False)
