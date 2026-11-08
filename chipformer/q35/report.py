"""Text prompts for the noise-floor checks of the text path (rv/textcheck.py, rv/chipcheck.py,
q35/mute.py's anti-sink fit, fx/fxio/text_paged.py), and how to encode them."""

import torch

PROMPTS = [
    ("chat", "Explain why the sky is blue in two sentences."),
    ("raw", 'def fibonacci(n):\n    """Return the n-th Fibonacci number."""\n'),
    ("chat", "What is 37 * 43? Think step by step."),
    ("chat", "Traduis en français : The weather is lovely today and I want to go for a walk."),
    ("chat", "写一首关于秋天的短诗。"),
    ("chat", "List three differences between TCP and UDP."),
    ("raw", "The CHIP-8 is a virtual machine from the 1970s that"),
    ("raw", "Q: If a train leaves at 3pm travelling 60 km/h, how far has it gone by 5:30pm?\nA:"),
    ("chat", "Write a haiku about a lighthouse."),
    ("chat", "Give me a Python one-liner that reverses the words in a sentence."),
    ("chat", "What causes the seasons on Earth?"),
    ("raw", "In 1969, Apollo 11"),
    ("chat", "Summarise the plot of Romeo and Juliet in three sentences."),
    ("chat", "Solve for x: 3x + 7 = 22."),
    ("raw", "SELECT name, COUNT(*) FROM orders"),
    ("chat", "¿Cuál es la capital de Australia y por qué no es Sídney?"),
]


def load(path, device, impl="sdpa"):
    from transformers import AutoModelForImageTextToText

    return AutoModelForImageTextToText.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation=impl).to(device).eval()


def encode(tok, kind, text):
    if kind == "chat":
        return tok.apply_chat_template([{"role": "user", "content": text}],
                                       add_generation_prompt=True, return_tensors="pt",
                                       enable_thinking=False)["input_ids"][0].tolist()
    return tok(text).input_ids
