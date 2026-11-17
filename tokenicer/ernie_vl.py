# Copyright 2026 ModelCloud.ai
# SPDX-License-Identifier: Apache-2.0
"""Native tokenizer metadata for the legacy Baidu ERNIE VL SentencePiece bundle.

Source schema: baidu/ERNIE-4.5-VL-28B-A3B-PT at
e3815e65c607ea211bfe21b46ab0cd264b76731c (Apache-2.0).
No checkpoint Python is imported. Transformers owns SentencePiece conversion.
"""

from transformers import PreTrainedConfig, PreTrainedTokenizerFast

MODEL_TYPES = {"ernie4_5_moe_vl", "ernie4_5_vl_moe"}
SPECIAL_TOKENS = {
    "image_token": "<|IMAGE_PLACEHOLDER|>",
    "image_start_token": "<|IMAGE_START|>",
    "image_end_token": "<|IMAGE_END|>",
    "video_token": "<|VIDEO_PLACEHOLDER|>",
    "video_start_token": "<|VIDEO_START|>",
    "video_end_token": "<|VIDEO_END|>",
}


def normalize_ernie_vl(tokenizer):
    tokens = dict(SPECIAL_TOKENS)
    for key in tokens:
        value = getattr(tokenizer, key, None)
        if value is not None:
            tokens[key] = value
    missing = {key: value for key, value in tokens.items() if key not in tokenizer.SPECIAL_TOKENS_ATTRIBUTES}
    if missing:
        tokenizer._set_model_specific_special_tokens(missing)
    tokenizer.add_special_tokens({"extra_special_tokens": list(tokens.values())}, replace_extra_special_tokens=False)
    def normalize_template(template):
        return template.replace("<|image@placeholder|>", tokenizer.image_token).replace(
            "<|video@placeholder|>", tokenizer.video_token
        )
    template = tokenizer.chat_template
    if isinstance(template, str):
        tokenizer.chat_template = normalize_template(template)
    elif isinstance(template, dict):
        tokenizer.chat_template = {key: normalize_template(value) for key, value in template.items()}
    return tokenizer


def load_ernie_vl(path, **kwargs):
    # get_tokenizer_config reads JSON only, including when auto_map is hostile.
    from transformers.models.auto.tokenization_auto import get_tokenizer_config

    config = kwargs.get("config")
    is_ernie = getattr(config, "model_type", None) in MODEL_TYPES
    if not is_ernie:
        try:
            metadata = get_tokenizer_config(path, **kwargs)
        except OSError:
            return None
        is_ernie = metadata.get("tokenizer_class") == "Ernie4_5_VLTokenizer"
    if not is_ernie:
        return None
    values = dict(kwargs)
    values.pop("config", None)
    values.pop("code_revision", None)
    values["trust_remote_code"] = False
    tokenizer = normalize_ernie_vl(PreTrainedTokenizerFast.from_pretrained(path, **values))
    tokenizer._tokenicer_native_ernie_vl = True
    return tokenizer


def ernie_vl_model_config(path, **kwargs):
    values, _ = PreTrainedConfig.get_config_dict(path, **kwargs)
    return PreTrainedConfig.from_dict(values)
