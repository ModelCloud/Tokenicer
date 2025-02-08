import logging
from typing import Union, List, Optional
from transformers import AutoConfig, PreTrainedTokenizer, PreTrainedModel, AutoTokenizer
from .util import candidate_ids
from .const import DEFAULT_PAD_TOKENS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Tokenicer:
    tokenizer: Union[str, PreTrainedTokenizer] = None
    trust_remote: bool = False

    @classmethod
    def load(cls, tokenizer_or_path: Union[str, PreTrainedTokenizer], trust_remote: bool = False):
        if tokenizer_or_path is None:
            raise ValueError("`tokenizer` cannot be None.")
        tokenicer = cls()
        tokenicer.trust_remote = trust_remote
        if isinstance(tokenizer_or_path, PreTrainedTokenizer):
            tokenicer.tokenizer = tokenizer_or_path
        elif isinstance(tokenizer_or_path, str):
            tokenicer.tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_path, trust_remote_code=trust_remote)
        else:
            raise ValueError(
                f"Unsupported type in tokenizer_or_path: {type(tokenizer_or_path)}. Expected str or PreTrainedTokenizer.")
        return tokenicer

    def auto_assign_pad_token(
        self,
        model_or_path: Union[str, PreTrainedModel],
        pad_tokens: Optional[List[Union[str, int]]] = None,
    ):
        if model_or_path is None:
            raise ValueError("`model_or_path` cannot be None.")

        if isinstance(model_or_path, str):
            config = AutoConfig.from_pretrained(model_or_path, trust_remote_code=self.trust_remote)
        elif isinstance(model_or_path, PreTrainedModel):
            config = getattr(model_or_path, "config", None)
        else:
            raise ValueError(f"Unsupported type in model_or_path: {type(model_or_path)}. Expected str or PreTrainedModel.")

        if config is None:
            raise ValueError("Could not retrieve config from the provided model_or_path.")

        pad_token_id = config.pad_token_id

        if pad_token_id is None:
            vocab = self.tokenizer.get_vocab()
            pad_tokens = pad_tokens or []
            pad_token_ids = candidate_ids(pad_tokens, vocab)
            default_candidate_ids = candidate_ids(DEFAULT_PAD_TOKENS, vocab)

            results = pad_token_ids + default_candidate_ids

            for candidate_id in results:
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

        self.tokenizer.pad_token_id = pad_token_id
        self.tokenizer.pad_token = self.tokenizer.decode([pad_token_id])

        logger.info(f"Assigned pad_token_id={pad_token_id} (token='{self.tokenizer.pad_token}').")
