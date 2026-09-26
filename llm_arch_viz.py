#!/usr/bin/env python
"""Visualize a Hugging Face transformers model architecture without downloading weights.

    python llm_arch_viz.py Qwen/Qwen2.5-7B-Instruct --out out/qwen2.5-7b.html
    python llm_arch_viz.py openai/privacy-filter --task token-classification
    python llm_arch_viz.py Qwen/Qwen2.5-7B-Instruct --json     # facts only, for checking a hypothesis

What it does
  1. AutoConfig.from_pretrained  -> config card (attention, positional encoding, norm, FFN, vocab, ctx)
  2. instantiate the model on torch's "meta" device (shapes only, zero bytes of weights)
  3. walk named_modules() -> module tree with parameter counts per module
  4. expand one decoder layer into a block diagram (attention -> residual -> MLP -> residual)
  5. estimate memory: weights at bf16/fp16/int4, KV cache per token, total at a given context
Output: one self-contained HTML file (inline SVG, no JS deps) plus a JSON summary on stdout with --json.
"""
from __future__ import annotations

import argparse
import html
import json
import math
import sys
from collections import OrderedDict
from dataclasses import dataclass, field

import torch
from transformers import AutoConfig

# --------------------------------------------------------------------------- facts from config

def _get(cfg, *names, default=None):
    for n in names:
        if hasattr(cfg, n) and getattr(cfg, n) is not None:
            return getattr(cfg, n)
    return default


def text_config(cfg):
    """Multimodal configs nest the LLM under text_config; return the LLM part and the top."""
    for n in ("text_config", "llm_config", "language_config"):
        sub = getattr(cfg, n, None)
        if sub is not None and hasattr(sub, "hidden_size"):
            return sub
    return cfg


def facts(cfg) -> "OrderedDict[str, object]":
    t = text_config(cfg)
    heads = _get(t, "num_attention_heads", "n_head", "num_heads")
    kv = _get(t, "num_key_value_heads", default=heads)
    hidden = _get(t, "hidden_size", "n_embd", "d_model")
    head_dim = _get(t, "head_dim", default=(hidden // heads if hidden and heads else None))
    layers = _get(t, "num_hidden_layers", "n_layer", "num_layers")
    if kv == heads:
        attn_kind = "MHA"
    elif kv == 1:
        attn_kind = "MQA"
    else:
        attn_kind = f"GQA {heads // kv}x"
    rope_params = _get(t, "rope_parameters", default=None) or {}
    if isinstance(rope_params, dict) and "rope_theta" not in rope_params and rope_params:
        rope_params = next(iter(rope_params.values())) if isinstance(next(iter(rope_params.values())), dict) else rope_params
    rope_theta = _get(t, "rope_theta", default=(rope_params.get("rope_theta") if isinstance(rope_params, dict) else None))
    rope_scaling = _get(t, "rope_scaling", default=(rope_params if isinstance(rope_params, dict) and rope_params.get("rope_type", "default") != "default" else None))
    if isinstance(rope_scaling, dict) and (rope_scaling.get("rope_type") or rope_scaling.get("type") or "default") == "default":
        rope_scaling = None
    partial = _get(t, "partial_rotary_factor")
    pos = "RoPE" if rope_theta or _get(t, "rotary_emb_base") else _get(t, "position_embedding_type", default="learned/absolute")
    if pos == "RoPE":
        pos = f"RoPE · θ={rope_theta:g}" if rope_theta else "RoPE"
        if partial:
            pos += f" · partial {partial:.0%}"
        if rope_scaling:
            pos += f" · scaling {rope_scaling.get('type') or rope_scaling.get('rope_type')}"
    norm_eps = _get(t, "rms_norm_eps", "layer_norm_eps", "layer_norm_epsilon")
    norm = "RMSNorm" if _get(t, "rms_norm_eps") else "LayerNorm"
    experts = _get(t, "num_experts", "num_local_experts", "n_routed_experts")
    top_k = _get(t, "num_experts_per_tok", "num_experts_per_token", "top_k")
    shared = _get(t, "n_shared_experts", "num_shared_experts")
    ffn = f"MoE {experts} experts · top-{top_k}" + (f" · +{shared} shared" if shared else "") if experts else "dense"
    inter = _get(t, "intermediate_size", "n_inner", "ffn_dim")
    act = _get(t, "hidden_act", "activation_function", "hidden_activation")
    sliding = _get(t, "sliding_window")
    layer_types = _get(t, "layer_types")
    out = OrderedDict()
    out["architecture"] = (cfg.architectures or ["?"])[0] if getattr(cfg, "architectures", None) else type(cfg).__name__
    out["model_type"] = cfg.model_type
    out["attention"] = f"{attn_kind} · heads {heads} · kv heads {kv} · head_dim {head_dim}" + (" · QK-norm" if _get(t, "use_qk_norm", "qk_norm", "qk_layernorm") else "")
    out["position"] = pos
    out["norm"] = f"{norm} · eps {norm_eps:g}" if norm_eps else norm
    out["ffn"] = f"{ffn} · inter {inter} · {act}"
    out["embedding"] = f"vocab {_get(t, 'vocab_size'):,} · tie {bool(_get(t, 'tie_word_embeddings', default=False))}"
    out["context"] = f"max_position_embeddings {_get(t, 'max_position_embeddings')}" + (f" · sliding window {sliding}" if sliding else "")
    if layer_types and len(set(layer_types)) > 1:
        out["layer_schedule"] = layer_types
    out["dims"] = {"hidden": hidden, "layers": layers, "heads": heads, "kv_heads": kv, "head_dim": head_dim,
                   "intermediate": inter, "vocab": _get(t, "vocab_size"), "ctx": _get(t, "max_position_embeddings")}
    return out

# --------------------------------------------------------------------------- module tree on meta device

@dataclass
class Node:
    name: str
    cls: str
    params: int = 0
    own: int = 0
    children: list = field(default_factory=list)
    shapes: list = field(default_factory=list)


def build_model(cfg, task: str):
    from transformers import (AutoModel, AutoModelForCausalLM, AutoModelForTokenClassification,
                              AutoModelForSequenceClassification)
    cls = {"causal-lm": AutoModelForCausalLM, "token-classification": AutoModelForTokenClassification,
           "sequence-classification": AutoModelForSequenceClassification, "base": AutoModel}[task]
    with torch.device("meta"):
        try:
            return cls.from_config(cfg, dtype=torch.bfloat16)
        except Exception as e:  # noqa: BLE001
            if task != "base":
                print(f"[warn] {cls.__name__} failed ({e.__class__.__name__}: {e}); falling back to AutoModel", file=sys.stderr)
                return AutoModel.from_config(cfg, dtype=torch.bfloat16)
            raise


def module_tree(model) -> Node:
    root = Node("model", type(model).__name__)
    index = {"": root}
    for name, mod in model.named_modules():
        if name == "":
            continue
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        parent = index[parent_name]
        n = Node(name.rsplit(".", 1)[-1], type(mod).__name__)
        n.own = sum(p.numel() for p in mod.parameters(recurse=False))
        n.shapes = [tuple(p.shape) for p in mod.parameters(recurse=False)]
        parent.children.append(n)
        index[name] = n
    # total params bottom-up
    def total(n: Node) -> int:
        n.params = n.own + sum(total(c) for c in n.children)
        return n.params
    total(root)
    return root


def collapse_repeats(n: Node) -> Node:
    """layers.0 … layers.N become one child 'layers.[0..N]' with a ×N badge (same class & param count)."""
    if not n.children:
        return n
    groups: "OrderedDict[tuple, list]" = OrderedDict()
    for c in n.children:
        key = (c.cls, c.params, c.name.rstrip("0123456789") if c.name[-1:].isdigit() else c.name)
        groups.setdefault(key, []).append(c)
    out = []
    for (cls, params, base), members in groups.items():
        first = collapse_repeats(members[0])
        if len(members) > 1 and all(m.name.rstrip("0123456789") == base for m in members):
            first.name = f"{base}[0..{len(members) - 1}]  ×{len(members)}"
        elif len(members) > 1:
            for m in members:
                out.append(collapse_repeats(m))
            continue
        out.append(first)
    n.children = out
    return n

# --------------------------------------------------------------------------- memory

def memory(f: "OrderedDict", total_params: int, ctx: int, batch: int = 1):
    d = f["dims"]
    kv_per_token = 2 * d["layers"] * d["kv_heads"] * d["head_dim"] * 2  # K and V, bf16
    return OrderedDict(
        params_total=total_params,
        weights_bf16_gib=total_params * 2 / 2**30,
        weights_int8_gib=total_params / 2**30,
        weights_int4_gib=total_params * 0.5 / 2**30,
        kv_cache_bytes_per_token=kv_per_token,
        kv_cache_gib_at_ctx=kv_per_token * ctx * batch / 2**30,
        ctx_used=ctx,
    )

# --------------------------------------------------------------------------- HTML / SVG rendering

def fmt(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def tree_html(n: Node, total: int, depth=0) -> str:
    pct = 100 * n.params / total if total else 0
    shape = ", ".join("×".join(map(str, s)) for s in n.shapes[:2])
    label = (f"<span class=nm>{html.escape(n.name)}</span> <span class=cls>{html.escape(n.cls)}</span> "
             f"<span class=pm>{fmt(n.params)}</span> <span class=pct>{pct:.1f}%</span>"
             + (f" <span class=shape>{html.escape(shape)}</span>" if shape else ""))
    bar = f"<span class=bar style='width:{max(pct, 0.3):.1f}%'></span>"
    if not n.children:
        return f"<li>{bar}{label}</li>"
    open_attr = " open" if depth < 2 else ""
    return f"<li><details{open_attr}><summary>{bar}{label}</summary><ul>" + "".join(
        tree_html(c, total, depth + 1) for c in n.children) + "</ul></details></li>"


def find_layer(root: Node):
    """The first repeated block (decoder layer) in the collapsed tree."""
    stack = [root]
    while stack:
        n = stack.pop(0)
        if "×" in n.name and n.children:
            return n
        stack.extend(n.children)
    return None


def layer_svg(layer: Node, dims: dict) -> str:
    """Block diagram of one decoder layer: each direct child is a box, in order, with residual arrows."""
    import re as _re
    order = [(r"input_layernorm|ln_1|pre_.*attn|attn.*_norm$|attention_norm", 0), (r"attn|attention", 1),
             (r"post_attention_layernorm|ln_2|pre_.*(mlp|ffn|feedforward)|ffn_norm|mlp_norm", 2),
             (r"mlp|moe|ffn|feed_forward|block_sparse|experts", 3), (r"post_(mlp|ffn|feedforward)", 4)]
    def rank(c: Node):
        key = (c.name + " " + c.cls).lower()
        for pat, r in order:
            if _re.search(pat, key):
                return r
        return 5
    kids = sorted(layer.children, key=rank)
    if not kids:
        return ""
    W, H, GAP, X0, Y0 = 340, 62, 26, 200, 40
    n = len(kids)
    height = Y0 + n * (H + GAP) + 30
    hidden = dims.get("hidden")
    parts = [f"<svg viewBox='0 0 640 {height}' xmlns='http://www.w3.org/2000/svg' font-family='ui-monospace,Menlo,monospace' font-size='12'>",
             "<defs><marker id='a' viewBox='0 0 10 10' refX='9' refY='5' markerWidth='7' markerHeight='7' orient='auto'><path d='M0 0L10 5L0 10z' fill='#64748b'/></marker></defs>",
             f"<text x='{X0}' y='22' font-size='13' fill='#334155'>hidden_states [B, S, {hidden}]</text>"]
    kinds = {"attn": "#fde68a", "mlp": "#bfdbfe", "moe": "#bfdbfe", "norm": "#e2e8f0", "expert": "#bfdbfe"}
    def color(c: Node):
        k = (c.cls + c.name).lower()
        for key, col in kinds.items():
            if key in k:
                return col
        return "#f1f5f9"
    for i, c in enumerate(kids):
        y = Y0 + i * (H + GAP)
        parts.append(f"<rect x='{X0}' y='{y}' width='{W}' height='{H}' rx='8' fill='{color(c)}' stroke='#94a3b8'/>")
        parts.append(f"<text x='{X0 + W / 2}' y='{y + 24}' text-anchor='middle' font-weight='600' fill='#0f172a'>{html.escape(c.cls)}</text>")
        parts.append(f"<text x='{X0 + W / 2}' y='{y + 40}' text-anchor='middle' fill='#475569'>{html.escape(c.name)} · {fmt(c.params)}</text>")
        shapes = [f"{g.name} {g.shapes[0][1]}→{g.shapes[0][0]}" for g in c.children if g.shapes and len(g.shapes[0]) == 2][:4]
        if shapes:
            parts.append(f"<text x='{X0 + W / 2}' y='{y + 53}' text-anchor='middle' fill='#7c3aed' font-size='10'>{html.escape(' · '.join(shapes))}</text>")
        if i < n - 1:
            parts.append(f"<line x1='{X0 + W / 2}' y1='{y + H}' x2='{X0 + W / 2}' y2='{y + H + GAP}' stroke='#64748b' marker-end='url(#a)'/>")
    # residual arrows: from before a norm to after the following block (norm -> block pairs)
    for i in range(0, n - 1):
        if "norm" in kids[i].cls.lower() and i + 1 < n:
            y_from = Y0 + i * (H + GAP) - GAP / 2
            y_to = Y0 + (i + 2) * (H + GAP) - GAP / 2 if i + 2 <= n else height - 20
            parts.append(f"<path d='M{X0 + W / 2} {y_from} H{X0 + W + 40} V{y_to} H{X0 + W / 2 + 4}' fill='none' stroke='#94a3b8' stroke-dasharray='4 3' marker-end='url(#a)'/>")
            parts.append(f"<text x='{X0 + W + 46}' y='{(y_from + y_to) / 2}' fill='#64748b' font-size='11'>+ residual</text>")
    parts.append("</svg>")
    return "".join(parts)


def render_html(model_id: str, f: "OrderedDict", root: Node, mem: "OrderedDict", top_shapes: list, hub_total=None) -> str:
    total = root.params
    layer = find_layer(root)
    d = f["dims"]
    rows = "".join(f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
                   for k, v in f.items() if k not in ("dims", "layer_schedule"))
    sched = ""
    if f.get("layer_schedule"):
        cells = "".join(f"<i class='{'g' if 'full' in t or 'global' in t else 'l'}' title='{i}: {t}'></i>" for i, t in enumerate(f["layer_schedule"]))
        sched = f"<div class=sched>{cells}</div><p class=muted>layer schedule · green = linear/sliding, blue = full attention</p>"
    dims_rows = "".join(f"<tr><th>{k}</th><td>{v:,}</td></tr>" for k, v in d.items() if isinstance(v, int))
    mem_rows = "".join(f"<tr><th>{k}</th><td>{(f'{v:,.2f}' if isinstance(v, float) else f'{v:,}')}</td></tr>" for k, v in mem.items())
    top = "".join(f"<tr><th>{html.escape(n)}</th><td>{fmt(p)}</td><td>{100 * p / total:.1f}%</td></tr>" for n, p in top_shapes)
    return f"""<!doctype html><html lang=en><meta charset=utf-8><title>{html.escape(model_id)} · architecture</title>
<style>
:root{{--ink:#0f172a;--mut:#64748b;--line:#e2e8f0;--bg:#fff;--card:#f8fafc}}
@media(prefers-color-scheme:dark){{:root{{--ink:#e2e8f0;--mut:#94a3b8;--line:#334155;--bg:#0b1220;--card:#111a2e}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 ui-sans-serif,system-ui,-apple-system,sans-serif}}
main{{max-width:1240px;margin:0 auto;padding:24px 16px;display:grid;grid-template-columns:360px 1fr;gap:24px}}
@media(max-width:900px){{main{{grid-template-columns:1fr}}}}
h1{{font-size:18px;margin:0 0 4px}} h2{{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);margin:20px 0 8px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}}
table{{border-collapse:collapse;width:100%}} th{{text-align:left;color:var(--mut);font-weight:500;padding:3px 8px 3px 0;white-space:nowrap;vertical-align:top}} td{{padding:3px 0;font-family:ui-monospace,Menlo,monospace;font-size:12.5px}}
.big{{font-size:26px;font-weight:700}} .muted{{color:var(--mut);font-size:12px;margin:4px 0 0}}
.sched{{display:flex;gap:2px;flex-wrap:wrap;margin-top:6px}} .sched i{{display:block;width:8px;height:18px;border-radius:2px;background:#22c55e}} .sched i.g{{background:#3b82f6}}
ul.tree{{list-style:none;padding-left:16px;margin:0}} ul.tree>li{{padding-left:0}} .tree li{{position:relative;padding:2px 0;font-family:ui-monospace,Menlo,monospace;font-size:12.5px}}
.tree summary{{cursor:pointer;list-style:none}} .tree summary::before{{content:'▸';display:inline-block;width:12px;color:var(--mut)}} details[open]>summary::before{{content:'▾'}}
.bar{{display:inline-block;height:9px;background:#93c5fd;border-radius:2px;margin-right:6px;vertical-align:middle;max-width:120px}}
.nm{{font-weight:600}} .cls{{color:var(--mut)}} .pm{{color:#2563eb;font-weight:600}} .pct{{color:var(--mut)}} .shape{{color:#7c3aed}}
svg{{width:100%;height:auto}}
</style>
<main>
<section>
<h1>{html.escape(model_id)}</h1><p class=muted>{html.escape(f['architecture'])} · meta device, no weights downloaded</p>
<div class=card><table>{rows}</table>{sched}</div>
<h2>Dimensions</h2><div class=card><table>{dims_rows}</table></div>
<h2>Parameters</h2><div class=card><div class=big>≈{fmt(total)}</div><p class=muted>{total:,} instantiated{(' · Hub safetensors ' + fmt(hub_total) + (' (matches)' if abs(hub_total - total) < total * 0.005 else f' (delta {fmt(abs(hub_total - total))}: checkpoint holds modules this architecture does not instantiate, e.g. an MTP head)')) if hub_total else ''}</p><table>{top}</table></div>
<h2>Memory (bf16 unless noted)</h2><div class=card><table>{mem_rows}</table><p class=muted>KV cache = 2 · layers · kv_heads · head_dim · 2 bytes per token</p></div>
</section>
<section>
<h2>One decoder layer{(' · ' + html.escape(layer.name)) if layer else ''}</h2><div class=card>{layer_svg(layer, d) if layer else '<p class=muted>no repeated block found</p>'}</div>
<h2>Module tree (click to expand)</h2><div class=card><ul class=tree>{tree_html(root, total)}</ul></div>
</section>
</main></html>"""

# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="HF model id or local path with config.json")
    ap.add_argument("--task", default="causal-lm", choices=["causal-lm", "token-classification", "sequence-classification", "base"])
    ap.add_argument("--out", help="HTML output path (default out/<model>.html)")
    ap.add_argument("--ctx", type=int, default=None, help="context length for the KV-cache estimate (default: min(max_position_embeddings, 32768))")
    ap.add_argument("--json", action="store_true", help="print the facts + memory summary as JSON")
    ap.add_argument("--trust-remote-code", action="store_true")
    a = ap.parse_args()

    cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=a.trust_remote_code)
    f = facts(cfg)
    model = build_model(cfg, a.task)
    root = module_tree(model)
    total = root.params
    ctx = a.ctx or min(f["dims"].get("ctx") or 32768, 32768)
    mem = memory(f, total, ctx)
    # top-level param split (before collapsing)
    def flat(n: Node, prefix=""):
        for c in n.children:
            yield prefix + c.name, c.params
    top_shapes = sorted(flat(root), key=lambda kv: -kv[1])[:6]
    # expand the biggest wrapper (e.g. `model`) one level so the split is embeddings / layers / norm / lm_head
    if root.children:
        inner = max(root.children, key=lambda c: c.params)
        if inner.children:
            top_shapes = sorted([(inner.name + "." + n, p) for n, p in flat(inner)] + [(c.name, c.params) for c in root.children if c is not inner], key=lambda kv: -kv[1])[:8]
    collapsed = collapse_repeats(root)

    # cross-check against the Hub's safetensors metadata (exact count of stored weights; differs when the
    # checkpoint carries modules the architecture does not instantiate, e.g. an MTP head)
    hub_total = None
    try:
        from huggingface_hub import model_info
        st = model_info(a.model, files_metadata=False).safetensors
        hub_total = getattr(st, "total", None)
    except Exception:  # noqa: BLE001
        pass
    summary = OrderedDict(model=a.model, facts=f, memory=mem, top_modules=top_shapes,
                          hub_safetensors_total=hub_total,
                          hub_delta=(hub_total - total) if hub_total else None)
    if a.json:
        print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    out = a.out or f"out/{a.model.replace('/', '__')}.html"
    import os
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(render_html(a.model, f, collapsed, mem, top_shapes, hub_total))
    check = f", hub safetensors {fmt(hub_total)} (delta {fmt(abs(hub_total - total))})" if hub_total else ""
    print(f"[ok] {a.model}: {total:,} params ({fmt(total)}){check}, bf16 {mem['weights_bf16_gib']:.1f} GiB, "
          f"KV {mem['kv_cache_bytes_per_token'] / 1024:.0f} KiB/token -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
