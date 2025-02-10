from collections import namedtuple

DEFAULT_PAD_TOKENS = [
        "<|finetune_right_pad_id|>",
        "<|pad|>",
        "<pad>",
        "<|unk|>",
        "<unk>"
]

TOKEN_TUPLE = namedtuple("TokenTuple", ["token", "token_id"])

MODEL_PAD_TOKEN_MAP = {
        "llama": TOKEN_TUPLE(token='<|finetune_right_pad_id|>', token_id=128004),
        "qwen2_5_vl": TOKEN_TUPLE(token='<|vision_pad|>', token_id=151654),
        "qwen2_vl": TOKEN_TUPLE(token='<|vision_pad|>', token_id=151654),
        "qwen2": TOKEN_TUPLE(token='<|fim_pad|>', token_id=151662),
}