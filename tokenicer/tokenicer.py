import logging
from typing import Union, List, Optional
from transformers import PreTrainedTokenizerBase, PreTrainedModel, AutoTokenizer
from .util import candidate_ids, retrieve_config_path, auto_config
from .const import DEFAULT_PAD_TOKENS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Tokenicer:
    tokenizer: Union[str, PreTrainedTokenizerBase] = None
    trust_remote: bool = False
    model_config = None

    @classmethod
    def load(cls, tokenizer_or_path: Union[str, PreTrainedTokenizerBase], trust_remote: bool = False):
        if tokenizer_or_path is None:
            raise ValueError("`tokenizer_or_path` cannot be None.")
        tokenicer = cls()
        tokenicer.trust_remote = trust_remote

        config_path = None
        if isinstance(tokenizer_or_path, PreTrainedTokenizerBase):
            tokenizer = tokenizer_or_path
            tokenicer.tokenizer = tokenizer
            config_path = retrieve_config_path(tokenizer)
        elif isinstance(tokenizer_or_path, str):
            tokenicer.tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_path, trust_remote_code=trust_remote)
            config_path = tokenizer_or_path
        else:
            raise ValueError(
                f"Unsupported type in tokenizer_or_path: {type(tokenizer_or_path)}. Expected str or PreTrainedTokenizerBase.")

        tokenicer.model_config = auto_config(config_path, trust_remote)

        if tokenicer.model_config is None:
            logger.warning(
                f"Cannot initialize `model_config` from the provided `tokenizer_or_path` parameter. "
                f"If, when calling the `auto_assign_pad_token()` API, it is necessary to specify the `model_or_path` parameter.",
            )

        return tokenicer

    def auto_assign_pad_token(
        self,
        model_or_path: Optional[Union[str, PreTrainedModel]] = None,
        pad_tokens: Optional[List[Union[str, int]]] = None,
    ):
        model_config = None
        if model_or_path is not None:
            if isinstance(model_or_path, str):
                model_config = auto_config(model_or_path, self.trust_remote)
            elif isinstance(model_or_path, PreTrainedModel):
                model_config = getattr(model_or_path, "config", None)
            else:
                raise ValueError(
                    f"Unsupported type in model_or_path: {type(model_or_path)}. Expected str or PreTrainedModel.")

            if model_config is None:
                raise ValueError("Could not retrieve config from the provided `model_or_path`.")
        else:
            if self.model_config is not None:
                model_config = self.model_config
            else:
                raise ValueError(
                    "Failed to initialize model config. Please ensure that the `Tokenicer.load(tokenizer_or_path)` parameter is set correctly."
                    "or set `auto_assign_pad_token(model_or_path)` parameter")

        pad_token_id = model_config.pad_token_id

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
            if isinstance(model_config.eos_token_id, list) and model_config.eos_token_id:
                pad_token_id = model_config.eos_token_id[0]
            else:
                pad_token_id = model_config.eos_token_id

        if pad_token_id is None:
            raise ValueError(
                "No valid pad token found. Please ensure you have set a valid `pad_tokens`."
            )

        self.tokenizer.pad_token_id = pad_token_id
        self.tokenizer.pad_token = self.tokenizer.decode([pad_token_id])

        logger.info(f"Assigned pad_token_id={pad_token_id} (token='{self.tokenizer.pad_token}').")
