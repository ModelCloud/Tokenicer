from typing import Union, List, Optional
from transformers import AutoConfig, PretrainedConfig


def candidate_ids(token_list: List[Union[str, int]], vocab: dict) -> List[Optional[int]]:
    token_ids = []
    for item in token_list:
        if isinstance(item, str):
            val = vocab.get(item)
            if val is not None:
                token_ids.append(val)
        if isinstance(item, int):
            if 0 <= item < len(vocab):
                token_ids.append(item)
    return token_ids


def candidate_id(token_list: List[Union[str, int]], vocab: dict) -> Optional[int]:
    token_ids = candidate_ids(token_list=token_list, vocab=vocab)
    return token_ids[0] if token_ids else None


def retrieve_config_path(obj) -> Optional[str]:
    path = getattr(obj, "name_or_path", None)
    if path:
        return path

    path = getattr(obj, "_name_or_path", None)
    if path:
        return path

    init_kwargs = getattr(obj, "init_kwargs", None)
    if init_kwargs and "name_or_path" in init_kwargs:
        return init_kwargs["name_or_path"]
    return None


def auto_config(path, trust_remote) -> Optional[PretrainedConfig]:
    config = AutoConfig.from_pretrained(path, trust_remote_code=trust_remote)
    model_config = None
    if isinstance(config, PretrainedConfig):
        model_config = config
    return model_config