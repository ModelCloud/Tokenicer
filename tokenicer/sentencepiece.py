# Copyright 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path

from tokenizers import Tokenizer, normalizers
from tokenizers.models import BPE
from .equivalence import transformers_class_name
from .serialized import normalize_legacy_metaspace
from .util import _tokenizer_metadata


_PIPELINE_OVERRIDES = frozenset({
    "legacy", "add_prefix_space", "tokenizer_object", "tokenizer_file", "vocab", "merges",
    "normalizer", "pre_tokenizer", "decoder",
})


def _sentencepiece_pipeline(metadata):
    if metadata.get("model", {}).get("type") != "BPE":
        return False
    pre_tokenizer = metadata.get("pre_tokenizer") or {}
    if pre_tokenizer.get("type") == "Metaspace" and pre_tokenizer.get("replacement") == "▁":
        return True
    replace = {"type": "Replace", "pattern": {"String": " "}, "content": "▁"}
    return metadata.get("normalizer") in (
        {"type": "Sequence", "normalizers": [replace]},
        {"type": "Sequence", "normalizers": [{"type": "Prepend", "prepend": "▁"}, replace]},
    )


def normalize_sentencepiece(
    tokenizer, path, *, historical_conversion=False, prefer_serialized_tokenizer=True, resolve_remote=False, **kwargs,
):
    """Preserve native Llama's SP conversion without changing its provider or token IDs."""
    # Exact class identity: Transformers' LlamaTokenizer or a registered verbatim copy of it.
    # Subclasses may change the pipeline, so they are never normalized.
    if transformers_class_name(type(tokenizer)) != "LlamaTokenizer" or _PIPELINE_OVERRIDES.intersection(kwargs):
        return
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        return

    vocab_file = getattr(tokenizer, "vocab_file", None)
    if not vocab_file and not path:
        return
    if vocab_file is not None and os.path.isfile(vocab_file):
        serialized = _tokenizer_metadata(os.path.dirname(vocab_file), "tokenizer.json")
    else:
        if not path or (not resolve_remote and not os.path.isdir(path)):
            return
        selection = {"local_files_only": True, **kwargs}
        serialized = _tokenizer_metadata(path, "tokenizer.json", **selection)
    if serialized:
        if prefer_serialized_tokenizer and _sentencepiece_pipeline(serialized):
            normalize_legacy_metaspace(serialized)
            source = Tokenizer.from_str(json.dumps(serialized))
            backend.normalizer = source.normalizer
            backend.pre_tokenizer = source.pre_tokenizer
            backend.decoder = source.decoder
        return

    if not historical_conversion or vocab_file is None or not os.path.isfile(vocab_file):
        return
    from sentencepiece import sentencepiece_model_pb2

    proto = sentencepiece_model_pb2.ModelProto()
    proto.ParseFromString(Path(vocab_file).read_bytes())
    spec = proto.normalizer_spec
    if (
        proto.trainer_spec.model_type != 2
        or spec.name != "identity"
        or spec.precompiled_charsmap
        or spec.remove_extra_whitespaces
        or not spec.escape_whitespaces
    ):
        return
    model = json.loads(backend.to_str())["model"]
    if model.pop("type") != "BPE":
        return
    vocab = {piece.piece: i for i, piece in enumerate(proto.pieces)}
    if model["vocab"] != vocab:
        return
    try:
        from transformers.tokenization_utils_base import generate_merges
    except ImportError:
        from transformers.convert_slow_tokenizer import generate_merges

    # Source fast conversion ranks merges by SentencePiece scores, not vocabulary IDs.
    model["merges"] = generate_merges(vocab, [(piece.piece, piece.score) for piece in proto.pieces])
    converted = BPE(**{key: value for key, value in model.items() if value is not None})
    converted_normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
    backend.model = converted
    backend.normalizer = converted_normalizer
    backend.pre_tokenizer = None
