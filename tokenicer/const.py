DEFAULT_PAD_TOKENS = [
        "<|finetune_right_pad_id|>",
        "<|pad|>",
        "<pad>",
        "<|unk|>",
        "<unk>"
]

MODEL_PAD_TOKEN_MAP = {
        "llama": "<|finetune_right_pad_id|>",
        "qwen2_5_vl": "<|vision_pad|>",
        "qwen2": "<|fim_pad|>",
}