from typing import Union, List, Optional


def get_candidate_ids(token_list: List[Union[str, int]], vocab: dict) -> List[Optional[int]]:
    candidate_ids = []
    for item in token_list:
        if isinstance(item, str):
            candidate_ids.append(vocab.get(item))
        elif isinstance(item, int):
            candidate_ids.append(item)
    return candidate_ids


def get_model_path(model_or_path):
    if hasattr(model_or_path, "name_or_path"):
        model_path = model_or_path.name_or_path
    elif hasattr(model_or_path, "config") and hasattr(model_or_path.config, "name_or_path"):
        model_path = model_or_path.config.name_or_path
    else:
        model_path = None
    return model_path