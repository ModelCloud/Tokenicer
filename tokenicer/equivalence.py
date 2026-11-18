# Copyright 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Explicit equivalence between vendored tokenizer classes and Transformers' own.

Tokenicer applies class-specific repairs (SentencePiece pipeline restoration, the
serialized ``tokenizer.json`` preference) only to Transformers' library classes:
remote/custom tokenizer code owns its own behavior. A library that vendors an exact
copy of a Transformers tokenizer class (for example to pin a Transformers ABI)
registers that copy here so it receives the same repairs. Registration is opt-in and
keyed by the class object; a class name alone never grants equivalence.
"""

from threading import Lock


_LOCK = Lock()
_EQUIVALENTS: dict[type, str] = {}


def register_transformers_equivalent(tokenizer_class: type, transformers_class_name: str | None = None) -> None:
    """Treat ``tokenizer_class`` as an exact copy of ``transformers.<transformers_class_name>``.

    ``transformers_class_name`` defaults to ``tokenizer_class.__name__``.
    """

    if not isinstance(tokenizer_class, type):
        raise TypeError(f"expected a tokenizer class, got {tokenizer_class!r}")
    name = transformers_class_name or tokenizer_class.__name__
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError(f"invalid Transformers tokenizer class name: {name!r}")
    with _LOCK:
        _EQUIVALENTS[tokenizer_class] = name


def transformers_class_name(tokenizer_class: type) -> str | None:
    """Name of the Transformers library class ``tokenizer_class`` is (or copies), else ``None``."""

    if (getattr(tokenizer_class, "__module__", "") or "").startswith("transformers."):
        return tokenizer_class.__name__
    return _EQUIVALENTS.get(tokenizer_class)


__all__ = ["register_transformers_equivalent", "transformers_class_name"]
