# Copyright 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Registered verbatim copies of Transformers tokenizers get the library-class repairs."""

import pytest
from transformers import LlamaTokenizer, PreTrainedTokenizerFast

from tokenicer import register_transformers_equivalent
from tokenicer.equivalence import _EQUIVALENTS, transformers_class_name
from tokenicer.sentencepiece import normalize_sentencepiece


def _vendored(name):
    """A class outside `transformers.*`, as a library vendoring Transformers would have."""

    return type(name, (LlamaTokenizer,), {"__module__": "vendor.models.llama.tokenization_llama"})


@pytest.fixture(autouse=True)
def _isolated_registry():
    saved = dict(_EQUIVALENTS)
    yield
    _EQUIVALENTS.clear()
    _EQUIVALENTS.update(saved)


def _probe(cls):
    """An uninitialized instance whose backend reads are counted (the gate's first step)."""

    counted = type(cls.__name__, (cls,), {
        "__module__": cls.__module__,
        "backend_reads": 0,
        "backend_tokenizer": property(lambda self: type(self)._read()),
        "_read": classmethod(lambda klass: setattr(klass, "backend_reads", klass.backend_reads + 1)),
    })
    return counted


def test_library_classes_resolve_by_their_own_name():
    assert transformers_class_name(LlamaTokenizer) == "LlamaTokenizer"
    # PreTrainedTokenizerFast is an alias of TokenizersBackend in Transformers 5.
    assert transformers_class_name(PreTrainedTokenizerFast) == PreTrainedTokenizerFast.__name__


def test_unregistered_copy_is_custom_code_even_with_the_same_name():
    assert transformers_class_name(_vendored("LlamaTokenizer")) is None


def test_registration_is_by_class_object_and_defaults_to_class_name():
    copy = _vendored("LlamaTokenizer")
    register_transformers_equivalent(copy)
    assert transformers_class_name(copy) == "LlamaTokenizer"
    assert transformers_class_name(_vendored("LlamaTokenizer")) is None


@pytest.mark.parametrize("bad", ["not a name", "1abc"])
def test_registration_rejects_invalid_names(bad):
    with pytest.raises(ValueError):
        register_transformers_equivalent(_vendored("X"), bad)


def test_registration_rejects_non_classes():
    with pytest.raises(TypeError):
        register_transformers_equivalent(object(), "LlamaTokenizer")


def test_sentencepiece_normalization_gate_follows_registration():
    copy = _probe(_vendored("LlamaTokenizer"))
    normalize_sentencepiece(object.__new__(copy), path=None)
    assert copy.backend_reads == 0

    register_transformers_equivalent(copy, "LlamaTokenizer")
    normalize_sentencepiece(object.__new__(copy), path=None)
    assert copy.backend_reads == 1


def test_copy_registered_as_another_class_is_not_normalized_as_llama():
    copy = _probe(_vendored("GemmaTokenizer"))
    register_transformers_equivalent(copy, "GemmaTokenizer")
    normalize_sentencepiece(object.__new__(copy), path=None)
    assert copy.backend_reads == 0
