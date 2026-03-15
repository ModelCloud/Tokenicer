# Copyright 2025 ModelCloud.ai
# Copyright 2025 qubitium@modelcloud.ai
# Contact: qubitium@modelcloud.ai, x.com/qubitium
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path


def _configure_hf_cache():
    # Keep remote-code modules and model cache writes inside a writable temp tree.
    hf_home = Path(os.environ.setdefault("HF_HOME", "/tmp/tokenicer_hf_home"))
    hf_hub_cache = Path(os.environ.setdefault("HF_HUB_CACHE", str(hf_home / "hub")))
    transformers_cache = Path(os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_hub_cache)))
    hf_modules_cache = Path(os.environ.setdefault("HF_MODULES_CACHE", str(hf_home / "modules")))

    for cache_dir in (hf_home, hf_hub_cache, transformers_cache, hf_modules_cache):
        cache_dir.mkdir(parents=True, exist_ok=True)


_configure_hf_cache()

from .tokenicer import Tokenicer

__all__ = ["Tokenicer"]
