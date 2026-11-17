<div align=center>


<img width="40%" alt="image" src="https://github.com/user-attachments/assets/af392964-8d3f-47a6-89e6-337743398051" />
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
* 10/15/2026 [0.0.16](https://github.com/ModelCloud/Tokenicer/releases/tag/v0.0.16): Keep the tokenization of the checkpoint's serialized `tokenizer.json` when a `*TokenizerFast` class resolves to a legacy tokenizer that rebuilds its own backend. Fixes `DeepSeek-R1-Distill-Llama-*` and other Llama-3-style checkpoints loaded from a local directory, where every word lost its leading whitespace.
* 07/22/2026 [0.0.14](https://github.com/ModelCloud/Tokenicer/releases/tag/v0.0.14): Auto-fix affected Mistral-family tokenizer regexes, including Laguna S 2.1.
* 03/03/2026 [0.0.7](https://github.com/ModelCloud/Tokenicer/releases/tag/v0.0.7): Fix Qwen 3.5 MoE compat.
* 02/09/2026 [0.0.6](https://github.com/ModelCloud/Tokenicer/releases/tag/v0.0.6): Fix ChatGLM compat. 
* 09/04/2025 [0.0.5](https://github.com/ModelCloud/Tokenicer/releases/tag/v0.0.5): Fix `pad_token_id` detection for `LongCat` model. 
* 02/21/2025 [0.0.4](https://github.com/ModelCloud/Tokenicer/releases/tag/v0.0.4): ⚡ Now `tokenicer` instance dynamically inherits the `native` `tokenizer.__class__` of tokenizer passed in or loaded via our `tokenicer.load()` api. CI now tests tokenizer compat from `64` different models.



* 02/10/2025 [0.0.2](https://github.com/ModelCloud/Tokenicer/releases/tag/v0.0.2): 🤗 Initial release!

## Features:

* Compatible with all HF `Transformers` recognized tokenizers
* Loads the tokenizer the checkpoint asks for: a checkpoint that declares a `*TokenizerFast` class keeps the tokenization of its serialized `tokenizer.json`, even when `Transformers` aliases that name to a legacy class that rebuilds its own backend (`LlamaTokenizer` hardcodes a sentencepiece `Metaspace` pre-tokenizer, which drops the leading whitespace of every word for Llama-3-style checkpoints). Pass `prefer_serialized_tokenizer=False` to keep the class `Transformers` resolves.
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

* If you already have a loaded or composite config, pass it directly so Tokenicer
  can normalize the resolved text config in-place:

```py
tokenizer = Tokenicer.load(tokenizer, model_config=model.config)
```

* Tokenizer classes are validated against the checkpoint's serialized `tokenizer.json`.
  Checkpoints that declare a `*TokenizerFast` class never fall back to a legacy class that
  tokenizes differently:

```py
tokenizer = Tokenicer.load('deepseek-ai/DeepSeek-R1-Distill-Llama-8B')
print(tokenizer('Hello world, this is a test.')['input_ids'])
# [9906, 1917, 11, 420, 374, 264, 1296, 13]

# Opt out and keep whatever class `Transformers` resolves for the checkpoint:
tokenizer = Tokenicer.load('deepseek-ai/DeepSeek-R1-Distill-Llama-8B', prefer_serialized_tokenizer=False)
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
