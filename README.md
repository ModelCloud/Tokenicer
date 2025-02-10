<h1 align="center">Toke(n)icer</h1>
<p align="center">A (nicer) tokenizer you want to use for model `inference` and `training`: with all known peventable `gotchas` normalized or auto-fixed.</p>
<p align="center">
    <a href="https://github.com/ModelCloud/Tokenicer/releases" style="text-decoration:none;"><img alt="GitHub release" src="https://img.shields.io/github/release/ModelCloud/Tokenicer.svg"></a>
    <a href="https://pypi.org/project/tokenicer/" style="text-decoration:none;"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/tokenicer"></a>
    <a href="https://pepy.tech/projects/tokenicer" style="text-decoration:none;"><img src="https://static.pepy.tech/badge/tokenicer" alt="PyPI Downloads"></a>
    <a href="https://github.com/ModelCloud/tokenicer/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/tokenicer"></a>
    <a href="https://huggingface.co/modelcloud/"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-ModelCloud-%23ff8811.svg"></a>
</p>

## News
* 02/10/2025 [0.0.1](https://github.com/ModelCloud/Tokenicer/releases/tag/v0.0.1): 🤗 Initial release!

## Features:

* Compatible with all `HF Transformers` compatible tokenizers
* Auto-fix `models` makers not setting or forgetting to set `padding_token`
* Auto-Fix `models` using and setting wrong `padding_token`: many `models` incorrectly use `eos_token` as `pad_token` which leads to subtle and hidden errors in post-training and inference when `batching` is used which is almost always.

## Upcoming Features:

* Add `automatic` tokenizer validation to `model` `training` and subsequence `inference` so that not only tokenizer config but actual `decode`/`encode` are 100% re-validated on model load. Often the case, `inference` and `training` engines modifies the traditional tokenizers causing subtle and inaccurate output when `inference` performed on a platform that is disjointed from the `trainer`. 

## Install

### PIP/UV 

```bash
pip install -v tokenicer
uv pip install -v tokenicer
```

### Install from source

```bash
# clone repo
git clone https://github.com/ModelCloud/Tokencier.git && cd Tokenicer

# compile
pip install -v . --no-build-isolation
```

## Use

* Use Tokenicer to replace AutoTokenizer, Tokenicer API is compatible with AutoTokenizer

```
from tokenicer import Tokenicer
tokenizer = Tokenicer.load(tokenizer_or_path='Qwen/Qwen2.5-0.5B-Instruct')

# Auto fix pad_token
tokenizer.auto_assign_pad_token()

bos_token = tokenizer.bos_token # <|im_end|>
pad_token = tokenizer.pad_token # <|fim_pad|>

text = "test string"
input_ids = tokenizer.encode(text, add_special_tokens=False) # [1944, 914]

```

## Citation

```
@misc{gptqmodel,
    author = {ModelCloud.ai and qubitium@modelcloud.ai},
    title = {Toke(n)icer},
    year = {2025},
    publisher = {GitHub},
    journal = {GitHub repository},
    howpublished = {\url{https://github.com/modelcloud/tokenicer}},
    note = {Contact: qubitium@modelcloud.ai}
}
