# Toke(n)icer

A (nicer) tokenizer you want to use for model `inference` and `training`: with all known peventable `gotchas` normalized or auto-healed.

## Features:

* Compatible with all `HF Transformers` compatible tokenizers
* Auto-fix `models` makers not setting or forgetting to set `padding_token`
* Auto-Fix `models` using and setting wrong `padding_token`: many `models` incorrectly use `eos_token` as `pad_token` which leads to subtle and hidden errors in post-training and inference when `batching` is used which is almost always.

## Upcoming Features:

* Add `automatic` tokenizer validation to `model` `training` and subsequence `inference` so that not only tokenizer config but actual `decode`/`encode` are 100% re-validated on model load. Often the case, `inference` and `training` engines modifies the traditional tokenizers causing subtle and inaccurate output when `inference` performed on a platform that is disjointed from the `trainer`. 


