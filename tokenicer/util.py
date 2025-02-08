from typing import Union, List, Optional


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