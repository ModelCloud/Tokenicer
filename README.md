<div align=center>


<img src='https://github.com/user-attachments/assets/3e103e67-784d-47b6-873d-88f7b0c886a7'>
<h1 align="center">Toke(n)icer</h1>
</div>

<p align="center">A (nicer) tokenizer you want to use for model inference and training: with all known peventable gotchas normalized or auto-fixed.</p>
<p align="center">
    <a href="https://github.com/ModelCloud/Tokenicer/releases" style="text-decoration:none;"><img alt="GitHub release" src="https://img.shields.io/github/release/ModelCloud/Tokenicer.svg"></a>
    <a href="https://pypi.org/project/tokenicer/" style="text-decoration:none;"><img alt="PyPI - Version" src="https://img.shields.io/pypi/v/tokenicer"></a>
    <a href="https://pepy.tech/projects/tokenicer" style="text-decoration:none;"><img src="https://static.pepy.tech/badge/tokenicer" alt="PyPI Downloads"></a>
    <a href="https://github.com/ModelCloud/tokenicer/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/tokenicer"></a>
    <a href="https://huggingface.co/modelcloud/"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-ModelCloud-%23ff8811.svg"></a>
</p>

## News
* 09/04/2025 [0.0.5](https://github.com/ModelCloud/Tokenicer/releases/tag/v0.0.5): Fixed `pad_token_id` fix for `LongCat` model. 
* 02/21/2025 [0.0.4](https://github.com/ModelCloud/Tokenicer/releases/tag/v0.0.4): ⚡ Now `tokenicer` instance dynamically inherits the `native` `tokenizer.__class__` of tokenizer passed in or loaded via our `tokenicer.load()` api. CI now tests tokenizer compat from `64` different models.



* 02/10/2025 [0.0.2](https://github.com/ModelCloud/Tokenicer/releases/tag/v0.0.2): 🤗 Initial release!

## Features:

* Compatible with all HF `Transformers` recognized tokenizers
* Auto-fix `models` not setting `padding_token`
* Auto-Fix `models` released with wrong `padding_token`: many `models` incorrectly use `eos_token` as `pad_token` which leads to subtle and hidden errors in post-training and inference when `batching` is used which is almost always.
* Zero external dependency outside of `Transformers`
  
## Upcoming Features:

* Add `automatic` tokenizer validation to `model` `training` and subsequent `inference` so that not only tokenizer config but actual `decode`/`encode` are 100% re-validated on model load. Often the case, `inference` and `training` engines modifies the traditional tokenizers causing subtle and inaccurate output when `inference` performed on a platform that is disjointed from the `trainer`. 

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
pip install -v . 
```

## Usage

* Replace all calls to `AutoTokenizer.from_pretrained()` with `Tokenizer.load()`: args are 100% compatible with `AutoTokenizer`

```py
# Replace `AutoTokenizer.from_pretrained()`
# from tokenizer import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')

# With `Tokenicer.load()`
from tokenicer import Tokenicer

# Returns `Tokenicer` instance that inherits original `Qwen2TokenizerFast` type.
tokenizer = Tokenicer.load('Qwen/Qwen2.5-0.5B-Instruct')

# That's it! Toke(n)icer has auto-fixed Qwen2.5-0.5B-Instruct's incorrect `pad_token`.
# Now this this model can be `trained` and `inferenced` correctly with `batch` and `masks`.
# Now use the new tokenizer like any normal HF PretrainedTokenizer(Fast)
print(f"pad_token: `{tokenizer.pad_token}`")
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
