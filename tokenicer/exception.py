class VerificationError(Exception):
    """Base class for all exceptions related to verification"""
    pass


class VerificationFileNotFoundError(VerificationError):
    def __init__(self):
        super().__init__("The verification file does not exist, please call the `save` API first.")


class VerificationInitializationError(VerificationError):
    def __init__(self, verify_json_path: str):
        super().__init__(f"Initialization verification data failed, please check {verify_json_path}.")


class ChatTemplateError(VerificationError):
    def __init__(self):
        super().__init__("Tokenizer does not support chat template.")


class ModelCofnfigNotFoundError(VerificationError):
    def __init__(self):
        super().__init__("Can not retrieve config path from the provided `pretrained_model_name_or_path`.")