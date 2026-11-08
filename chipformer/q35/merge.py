"""Widen a Qwen3.5 checkpoint and append chip layers, keeping Qwen's own computation intact.

Qwen's matrices are zero-padded to the new hidden size, so its dims never read or write chip
dims. The RMSNorms that see the full residual (input, post-attention, final) average over more
dims after widening; their gains are scaled by sqrt(D / D') so Qwen's normalized inputs stay the
same up to one bf16 rounding. The MTP head is dropped: HF ignores it, and its shapes would change.
"""

import copy
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open

TEXT = "model.language_model."
# which axis of each weight is the hidden dim, by name
IN_COLS = re.compile(
    r"(in_proj_(qkv|z|a|b)|q_proj|k_proj|v_proj|gate_proj|up_proj)\.weight$|embed_tokens\.weight$"
)
OUT_ROWS = re.compile(r"(out_proj|o_proj|down_proj)\.weight$|visual\.merger\.linear_fc2\.(weight|bias)$")
WIDE_NORMS = re.compile(r"(input_layernorm|post_attention_layernorm)\.weight$|language_model\.norm\.weight$")

IM_START = 248045  # <|im_start|> in Qwen3.5's tokenizer


def load(path):
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    sd = {}
    for f in sorted(path.glob("*.safetensors")):
        with safe_open(f, "pt") as fh:
            for k in fh.keys():  # noqa: SIM118 - safe_open is not iterable
                if not k.startswith("mtp."):
                    sd[k] = fh.get_tensor(k)
    return cfg, sd


def widen(cfg, sd, hidden, n_chip_layers=0):
    """Return (config, state_dict) with hidden size `hidden` and n zero chip layers appended."""
    tc = cfg["text_config"]
    d = tc["hidden_size"]
    assert hidden >= d
    scale = (d / hidden) ** 0.5
    out = {}
    for k, w in sd.items():
        if IN_COLS.search(k):
            w = torch.cat([w, w.new_zeros(*w.shape[:-1], hidden - d)], -1)
        elif OUT_ROWS.search(k):
            w = torch.cat([w, w.new_zeros(hidden - d, *w.shape[1:])], 0)
        elif WIDE_NORMS.search(k):
            g = (1.0 + w.float()) * scale - 1.0  # zero-centred gain: (1 + w)
            w = torch.cat([g.to(w.dtype), w.new_zeros(hidden - d)])
        out[k] = w
    cfg = copy.deepcopy(cfg)
    tc = cfg["text_config"]
    tc["hidden_size"] = hidden
    cfg["vision_config"]["out_hidden_size"] = hidden
    tc["mtp_num_hidden_layers"] = 0
    tc.pop("full_attention_interval", None)
    n0 = tc["num_hidden_layers"]
    tc["layer_types"] = tc["layer_types"] + ["linear_attention"] * n_chip_layers
    tc["num_hidden_layers"] = n0 + n_chip_layers
    for i in range(n0, n0 + n_chip_layers):
        out.update(empty_chip_layer(tc, i, out[f"{TEXT}layers.0.input_layernorm.weight"].dtype))
    return cfg, out


def empty_chip_layer(tc, i, dtype):
    """Zero weights for a linear-attention layer: on text tokens it adds exactly 0."""
    h, nk, nv = tc["hidden_size"], tc["linear_num_key_heads"], tc["linear_num_value_heads"]
    dk, dv, inter = tc["linear_key_head_dim"], tc["linear_value_head_dim"], tc["intermediate_size"]
    p = f"{TEXT}layers.{i}."
    z = lambda *s: torch.zeros(*s, dtype=dtype)
    return {
        p + "input_layernorm.weight": z(h),
        p + "post_attention_layernorm.weight": z(h),
        p + "linear_attn.in_proj_qkv.weight": z(2 * nk * dk + nv * dv, h),
        p + "linear_attn.in_proj_z.weight": z(nv * dv, h),
        p + "linear_attn.in_proj_a.weight": z(nv, h),
        p + "linear_attn.in_proj_b.weight": z(nv, h),
        p + "linear_attn.conv1d.weight": z(2 * nk * dk + nv * dv, 1, tc["linear_conv_kernel_dim"]),
        p + "linear_attn.dt_bias": z(nv),
        p + "linear_attn.A_log": z(nv),
        p + "linear_attn.norm.weight": torch.ones(dv, dtype=dtype),
        p + "linear_attn.out_proj.weight": z(h, nv * dv),
        p + "mlp.gate_proj.weight": z(inter, h),
        p + "mlp.up_proj.weight": z(inter, h),
        p + "mlp.down_proj.weight": z(h, inter),
    }


def instantiate(cfg, sd, dtype=torch.bfloat16):
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    config = Qwen3_5Config(**{k: v for k, v in cfg.items() if k not in ("architectures", "model_type")})
    with torch.device("meta"):
        model = Qwen3_5ForConditionalGeneration(config)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    missing = [k for k in missing if k != "lm_head.weight"]  # tied to embed_tokens
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    model.tie_weights()
    # rotary tables are non-persistent buffers, still on the meta device: rebuild them
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q

    inner = model.model
    inner.language_model.rotary_emb = q.Qwen3_5TextRotaryEmbedding(config=config.text_config)
    inner.visual.rotary_pos_emb = q.Qwen3_5VisionRotaryEmbedding(config.vision_config)
    return model.to(dtype).eval()


# --- the merged checkpoint -----------------------------------------------------------------

TAU_TEXT = 2.0**-40  # tau in every text token's embedding row
TAU_GAIN = 2.0**60  # final-norm gain on tau: chip logits at text positions ~ -2^20 / rms


def pick_hidden(base: int, n_chip: int) -> tuple[int, int]:
    """Smallest (hidden, m) with hidden = m * 4^k >= base + m + n_chip, a multiple of 128, and
    3 <= m <= 64 ballast dims (m * B^2 / hidden must be a power of 4)."""
    best = None
    for m in range(3, 65):
        k = 0
        while m * 4**k < base + m + n_chip or (m * 4**k) % 128:
            k += 1
        h = m * 4**k
        if best is None or h < best[0] or (h == best[0] and m > best[1]):
            best = (h, m)
    return best
