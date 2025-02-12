from typing import List, Optional
from dataclasses import dataclass, field


@dataclass
class VerifyData:
    format: str = 'simple'
    input: str = ''
    output: List[int] = field(default_factory=list)

    def __init__(self, input: str, output: List[int], format: str = 'simple'):
        self.format = format
        self.input = input
        self.output = output


@dataclass
class VerifyMeta:
    validator: str
    url: str

    def __init__(self, validator, url):
        self.validator = validator
        self.url = url


@dataclass
class VerifyConfig:
    meta: Optional[VerifyMeta] = None
    datasets: List[VerifyData] = field(default_factory=list)

    def __init__(self, datasets: List[VerifyData], meta: VerifyMeta = None):
        if meta is None:
            from .version import __version__
            meta = VerifyMeta(validator=f"tokenicer:{__version__}", url='https://github.com/ModelCloud/Tokenicer')
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
            'datasets': dataset_dict
        }

    @classmethod
    def from_dict(cls, data: dict):
        try:
            datasets_data = data.get('datasets')
            datasets = []
            if datasets_data is not None and isinstance(datasets_data, list):
                for data_item in datasets_data:
                    if isinstance(data_item, dict):
                        input = data_item.get('input')
                        output = data_item.get('output')
                        format = data_item.get('format')
                        if input is not None and output is not None and format is not None:
                            datasets.append(VerifyData(input=input, output=output, format=format))

            meta_data = data.get('meta')
            meta = None

            if meta_data is not None:
                validator = meta_data.get('validator')
                url = meta_data.get('url')

                if validator is not None and url is not None:
                    meta = VerifyMeta(validator=validator, url=url)

            return cls(datasets=datasets, meta=meta)

        except Exception as e:
            return None
