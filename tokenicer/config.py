from typing import List


class VerifyData:
    format: str
    input: str
    output: List[int]

    def __init__(self, input: str, output: List[int], format: str = 'simple'):
        self.format = format
        self.input = input
        self.output = output


class VerifyMeta:
    validator: str
    url: str

    def __init__(self, version, url):
        self.validator = version
        self.url = url


class VerifyConfig:
    meta: VerifyMeta
    datasets: List[VerifyData]

    def __init__(self, datasets: List[VerifyData], meta: VerifyMeta = None):
        if meta is None:
            from .version import __version__
            meta = VerifyMeta(version=__version__, url='https://github.com/ModelCloud/Tokenicer')
        self.meta = meta
        self.datasets = datasets

    def to_dict(self):
        dataset_dict = [
            {
                'format': data.format,
                'input': data.input,
                'output': data.output,
            } for data in self.datasets
        ]

        meta_dict = {
            'validator': self.meta.validator,
            'url': self.meta.url
        }

        return {
            'meta': meta_dict,
            'dataset': dataset_dict
        }
