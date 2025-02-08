import logging
from typing import Union, List, Optional
from transformers import AutoConfig, PreTrainedTokenizer, PreTrainedModel, AutoTokenizer
from .util import get_candidate_ids, get_model_path

logger = logging.getLogger(__name__)

class Tokenicer:
    DEFAULT_PAD_TOKENS = [
        "<|finetune_right_pad_id|>",
        "<|pad|>",
        "<pad>",
        "<|unk|>",
        "<unk>"
    ]

    @classmethod
    def auto_assign_pad_token(
        cls,
        model_or_path: Union[str, PreTrainedModel],
        tokenizer: PreTrainedTokenizer,
        user_pad_tokens: Optional[List[Union[str, int]]] = None,
        trust_remote: bool = False
    ) -> PreTrainedTokenizer:
        if model_or_path is None:
            raise ValueError("`model_or_path` cannot be None.")

        tokenizer_path = None
        if isinstance(model_or_path, str):
            config = AutoConfig.from_pretrained(model_or_path, trust_remote_code=trust_remote)
            if tokenizer is None:
                tokenizer_path = model_or_path
        elif isinstance(model_or_path, PreTrainedModel):
            config = getattr(model_or_path, "config", None)
            if tokenizer is None:
                tokenizer_path = get_model_path(model_or_path)
        else:
            raise ValueError(f"Unsupported type in user_pad_tokens: {type(model_or_path)}. Expected str or PreTrainedModel.")

        if tokenizer is None and tokenizer_path is not None:
            tokenizer = AutoTokenizer.from_pretrained(model_or_path, trust_remote_code=trust_remote)
        else:
            raise ValueError("`tokenizer` cannot be None.")

        if config is None:
            raise ValueError("Could not retrieve config from the provided model_or_path.")

        pad_token_id = config.pad_token_id

        if pad_token_id is None:
            vocab = tokenizer.get_vocab() if tokenizer else {}
            user_pad_tokens = user_pad_tokens or []
            user_candidate_ids = get_candidate_ids(user_pad_tokens, vocab)
            default_candidate_ids = get_candidate_ids(cls.DEFAULT_PAD_TOKENS, vocab)
            candidate_ids = user_candidate_ids + default_candidate_ids

            for candidate_id in candidate_ids:
                if candidate_id is not None:
                    pad_token_id = candidate_id
                    break

        if pad_token_id is None:
            if isinstance(config.eos_token_id, list) and config.eos_token_id:
                pad_token_id = config.eos_token_id[0]
            else:
                pad_token_id = config.eos_token_id

        if pad_token_id is None:
            raise ValueError(
                "No valid pad token found. Please ensure you have set a valid `pad_token_id`."
            )

        tokenizer.pad_token_id = pad_token_id
        tokenizer.pad_token = tokenizer.decode([pad_token_id])

        logger.info(f"Assigned pad_token_id={pad_token_id} (token='{tokenizer.pad_token}').")

        return tokenizer
