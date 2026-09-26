# llm-arch-viz

**Live:** https://kaihuan-huang.github.io/llm-arch-viz/ (cards for Qwen2.5-7B-Instruct, openai/privacy-filter, Qwen3.5-9B, Qwen3.8-27B, granite-embedding-278m)

Architecture card + module graph for any Hugging Face transformers model, without downloading weights.
The model is instantiated on torch's `meta` device (shapes only), so a 70B model takes seconds and zero disk.

```sh
.venv/bin/python llm_arch_viz.py Qwen/Qwen2.5-7B-Instruct            # -> out/Qwen__Qwen2.5-7B-Instruct.html
.venv/bin/python llm_arch_viz.py Qwen/Qwen2.5-7B-Instruct --json     # facts + memory as JSON (for checking a hypothesis)
.venv/bin/python llm_arch_viz.py openai/privacy-filter --task token-classification
.venv/bin/python llm_arch_viz.py ibm-granite/granite-embedding-278m-multilingual --task base
```

What the page shows: attention kind (MHA/GQA/MQA, heads, kv heads, head_dim, QK-norm), positional encoding,
norm, FFN (dense or MoE experts/top-k), vocab, context, layer schedule (if the config has one), parameter total
and split, memory (bf16/int8/int4 weights, KV-cache bytes per token and at a context length), one decoder layer as a
block diagram, and the full module tree with per-module parameter counts and weight shapes.

Setup: `uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python torch transformers huggingface_hub`.

The parameter total is cross-checked against the Hub's safetensors metadata; a delta means the checkpoint carries modules the architecture does not instantiate (for the Qwen3.5/3.8 family, the multi-token-prediction head). Memory figures are arithmetic, not measurements. MIT license.
