from collections import namedtuple

DEFAULT_PAD_TOKENS = [
        "<|finetune_right_pad_id|>",
        "<|pad|>",
        "<pad>",
        "<|unk|>",
        "<unk>"
]

PAD_TOKEN_TUPLE = namedtuple("PadTokenTuple", ["pad_token", "pad_token_id"])

MODEL_PAD_TOKEN_MAP = {
        "llama": PAD_TOKEN_TUPLE(pad_token='<|finetune_right_pad_id|>', pad_token_id=128004),
        "qwen2_5_vl": PAD_TOKEN_TUPLE(pad_token='<|vision_pad|>', pad_token_id=151654),
        "qwen2": PAD_TOKEN_TUPLE(pad_token='<|fim_pad|>', pad_token_id=151662),
}