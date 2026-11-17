# Copyright 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import tempfile
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE
from transformers.tokenization_utils_base import get_fast_tokenizer_file

from .util import _tokenizer_file, _tokenizer_metadata


def normalize_legacy_metaspace(metadata):
    """Translate only legacy boolean-only Metaspace nodes in tokenizer pipelines."""
    def visit(node):
        if not isinstance(node, dict):
            return False
        changed = False
        if node.get("type") == "Metaspace" and "prepend_scheme" not in node and "add_prefix_space" in node:
            prefix = node["add_prefix_space"]
            if not isinstance(prefix, bool):
                raise ValueError("Tokenicer: legacy Metaspace add_prefix_space must be a boolean")
            node["prepend_scheme"] = "always" if prefix else "never"
            del node["add_prefix_space"]
            changed = True
        elif node.get("type") == "Sequence":
            for field in ("pretokenizers", "decoders"):
                for child in node.get(field, []):
                    changed = visit(child) or changed
        return changed

    pre = visit(metadata.get("pre_tokenizer"))
    decoder = visit(metadata.get("decoder"))
    return pre or decoder


def serialized_tokenizer_file(path, **kwargs):
    if "tokenizer_file" in kwargs:
        return kwargs["tokenizer_file"]
    if not isinstance(path, (str, os.PathLike)):
        return None
    if os.path.isfile(path):
        return os.fspath(path) if os.fspath(path).endswith(".json") else None
    metadata = _tokenizer_metadata(path, "tokenizer_config.json", **kwargs)
    filename = get_fast_tokenizer_file(metadata["fast_tokenizer_files"]) if "fast_tokenizer_files" in metadata else "tokenizer.json"
    return _tokenizer_file(path, filename, **kwargs)


def prepare_serialized_tokenizer(path, **kwargs):
    if "tokenizer_file" in kwargs or "tokenizer_object" in kwargs or kwargs.get("gguf_file") is not None:
        return None
    source = serialized_tokenizer_file(path, **kwargs)
    if source is None:
        return None
    metadata = json.loads(Path(source).read_bytes())
    if not isinstance(metadata, dict):
        raise ValueError(f"Tokenicer: serialized tokenizer must be an object: {source}")
    if not normalize_legacy_metaspace(metadata):
        return None
    content = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    directory = Path(kwargs.get("cache_dir") or HF_HUB_CACHE) / "tokenicer" / "metaspace-v1"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{digest}.json"
    try:
        if target.read_bytes() == content:
            return str(target)
    except FileNotFoundError:
        pass
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, prefix=f".{digest}-", delete=False) as stream:
            temporary = stream.name
            stream.write(content)
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return str(target)
