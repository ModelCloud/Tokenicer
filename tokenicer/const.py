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

from collections import namedtuple

DEFAULT_PAD_TOKENS = [
        "<|finetune_right_pad_id|>",
        "<|pad|>",
        "<pad>",
        "<|unk|>",
        "<unk>"
]

TOKEN_TUPLE = namedtuple("TokenTuple", ["token", "token_id"])

MODEL_PAD_TOKEN_MAP = {
        "llama": TOKEN_TUPLE(token='<|finetune_right_pad_id|>', token_id=128004),
        "qwen2_5_vl": TOKEN_TUPLE(token='<|vision_pad|>', token_id=151654),
        "qwen2_vl": TOKEN_TUPLE(token='<|vision_pad|>', token_id=151654),
        "qwen2": TOKEN_TUPLE(token='<|fim_pad|>', token_id=151662),
        "deepseek_v3": TOKEN_TUPLE(token='<｜▁pad▁｜>', token_id=2),
        "mpt": TOKEN_TUPLE(token='<|padding|>', token_id=1)
}

VERIFY_JSON_FILE_NAME = "tokenizer_verify.jsonl"
VERIFY_ENCODE_PARAMS = {"return_tensors": "pt", "add_special_tokens": False}

INPUT_KEY = "input"
TENSOR_KEY = "tensor"

VERIFY_DATASETS = [
        # English
        "Sure! I'd be happy to help. What kind of writing prompt are you looking for?",
        "Certainly! A comma (,) is used to separate items in a list, e.g., 'I bought apples, bananas, and oranges.' A semicolon (;) links related independent clauses, e.g., 'I have a meeting tomorrow; I need to prepare.' A colon (:) introduces a list or explanation, e.g., 'Here are the items you need: pen, paper, and ink.'",
        "Let's do it:\n\n1. 3.14159265359 + 2.71828182846 = 5.85987448205\n2. 5.6 * 2.3 = 12.88\n3. The square root of 123.456 is approximately 11.1111047355\n\nWould you like to explore more complex calculations? I can also work with exponents (e.g., 2^10 or 5.1^3.2).",
        "Let's break it down:\n\n1. **Balancing the chemical equation:** The unbalanced equation is: \n   H₂ + O₂ → H₂O. To balance it, we need 2 molecules of H₂ and 1 molecule of O₂ to form 2 molecules of H₂O: \n   **2H₂ + O₂ → 2H₂O.**\n\n2. **Area of a circle:** The formula for the area of a circle is \( A = \pi r^2 \). With a radius of 5.7 cm, the area is approximately: \n   \( A = 3.14159 \times (5.7)^2 = 102.041 \, \text{cm}^2.\)\n\n3. **Molar mass of NaCl:** Sodium chloride (NaCl) consists of one sodium (Na) atom and one chlorine (Cl) atom. The atomic masses are approximately: \n   Na = 22.99 g/mol, Cl = 35.45 g/mol. So, the molar mass of NaCl is: \n   **22.99 g/mol + 35.45 g/mol = 58.44 g/mol.**",
        # Chinese
        "在一个清晨，阳光透过窗帘缝隙洒在床单上，空气里弥漫着刚煮好的咖啡香。街道还很安静，偶尔有几只鸟儿在枝头跳跃。",
        "2025年，科技的发展速度令人惊叹！\n量子计算机的计算能力已达到10¹⁰次操作每秒，\n而ChatGPT模型的推理速度是传统计算机的100倍以上。\n公式E=mc²揭示了质量和能量的关系。\n今天的任务包括：\n1. 完成项目报告\n2. 参加9:00的会议\n3. 下午2:00开始的代码审查\n别忘了，创新与效率是成功的关键！",
        "2025年，科技的發展速度讓人驚訝！\n量子電腦的計算能力已達到 10¹⁰ 次操作每秒，\n而ChatGPT模型的推理速度是傳統電腦的100倍以上。\n例如，愛因斯坦的著名公式 E = mc²，\n揭示了質量和能量之間的關係。\n化學中，水的化學式 H₂O 代表著每個分子包含兩個氫原子和一個氧原子。\n今日的工作清單如下：\n1. 完成數學模型的推導：x² + 3x - 4 = 0\n2. 實驗室研究化學反應：2H₂ + O₂ → 2H₂O\n3. 進行下午3:00的會議\n每一步，都是知識積累的過程。",
        # Franch
        "Le matin, lorsque le soleil se lève lentement à l'horizon, la ville semble encore endormie. Les rues sont calmes, seules quelques personnes marchent rapidement pour commencer leur journée. L'air est frais, et les arbres, bien que dépouillés de leurs feuilles en hiver, semblent toujours veiller sur la ville. J'aime prendre un moment pour observer ce silence paisible avant que le bruit de la journée ne commence à envahir l'espace. Parfois, il suffit de quelques instants pour se reconnecter à soi-même et à l'instant présent.",
        # German
        "In der modernen Softwareentwicklung ist es wichtig, effizienten Code zu schreiben. Zum Beispiel kann ein einfacher `for`-Loop in Python wie folgt aussehen: ```python\nfor i in range(10):\n    print(i)\n``` Dieser Code gibt die Zahlen von 0 bis 9 aus. Es ist auch entscheidend, den Code so zu optimieren, dass er sowohl lesbar als auch schnell ist. Ein gut strukturierter Code trägt zu einer besseren Wartbarkeit bei und reduziert die Wahrscheinlichkeit von Fehlern.",
        # Spanish
        "# Este es un ejemplo de código en Python\ndef saludar(nombre):\n    print(f\"¡Hola, {nombre}!\")\n\n# Llamada a la función\nsaludar(\"Juan\")",
        # Arabic
        "الكيمياء هي دراسة المادة وتفاعلاتها. وتشمل العديد من الفروع مثل الكيمياء العضوية وغير العضوية، والكيمياء التحليلية والكيمياء الفيزيائية. تلعب الكيمياء دوراً مهماً في العديد من الصناعات مثل صناعة الأدوية، والبترول، والطاقة."
]