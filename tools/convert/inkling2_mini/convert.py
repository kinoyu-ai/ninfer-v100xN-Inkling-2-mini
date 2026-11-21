#!/usr/bin/env python3
"""Convert thinkingmachines/Inkling-2-Mini (HF safetensors + optional mtp.safetensors)
into a native lossless NInfer v2 `.ninfer` artifact (`inkling2-mini/nvfp4-mxfp8`).

All NVFP4 codes/scales/amax/denominators (layers 0..19), MXFP8 codes/UE8M0 scales
(layers 20..23), and BF16/FP32 tensors are bit-preserved with zero requantization.
Expert banks are organized into two contiguous TP2 intermediate halves `[2, 128, ...]`
so both TP1 (whole tensor) and TP2 (single contiguous axis-0 slice per GPU) load via
direct I/O with minimal H2D copy overhead.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.artifact.container import (
    ArtifactIdentity,
    ArtifactWriter,
    RAW_BYTES_V1,
    ResourceSpec,
    TensorSpec,
)

LAYOUT = "contiguous-le-v1"

LOCAL_LAYERS = {0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22}
FULL_LAYERS = {5, 11, 17, 23}
CORE_LAYERS = set(range(6, 18))
NVFP4_LAYERS = set(range(0, 20))
MXFP8_LAYERS = set(range(20, 24))


def _tensor_bytes(t: torch.Tensor) -> bytes:
    c = t.contiguous()
    if c.dtype == torch.bfloat16:
        arr = c.view(torch.uint16).numpy()
    elif c.dtype == torch.float32:
        arr = c.numpy()
    elif c.dtype == torch.uint8:
        arr = c.numpy()
    elif c.dtype == torch.float8_e4m3fn:
        arr = c.view(torch.uint8).numpy()
    else:
        raise ValueError(f"Unsupported tensor dtype: {c.dtype}")
    if sys.byteorder != "little" and arr.dtype.itemsize > 1:
        arr = arr.byteswap()
    return arr.tobytes()


def _require_tensor(sf, name: str, expected_dtype: torch.dtype, expected_shape: tuple[int, ...]) -> torch.Tensor:
    t = sf.get_tensor(name)
    if t.dtype != expected_dtype or tuple(t.shape) != expected_shape:
        raise ValueError(
            f"{name}: expected {expected_dtype} {expected_shape}, got {t.dtype} {tuple(t.shape)}"
        )
    return t


def _validate_config(model_dir: Path) -> None:
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    tc = cfg["text_config"]
    checks = {
        "hidden_size": 2304,
        "num_hidden_layers": 24,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "d_rel": 16,
        "rel_extent": 1024,
        "sliding_window_size": 1024,
        "vocab_size": 201024,
        "unpadded_vocab_size": 200058,
        "n_routed_experts": 128,
        "num_experts_per_tok": 8,
        "intermediate_size": 1152,
        "n_layers_per_attn_res_block": 6,
    }
    for key, expected in checks.items():
        if tc.get(key) != expected:
            raise ValueError(f"config.json text_config.{key}: expected {expected}, got {tc.get(key)}")
    loop = tc["loop_config"]
    if loop != {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64, "is_varloop": False}:
        raise ValueError(f"Unexpected loop_config: {loop}")


def build_inventory(model_dir: Path, include_mtp: bool) -> list[TensorSpec | ResourceSpec]:
    specs: list[TensorSpec | ResourceSpec] = []

    # Frontend resources
    for fname in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
    ):
        p = model_dir / fname
        if not p.is_file():
            raise FileNotFoundError(f"Missing frontend resource: {p}")
        specs.append(ResourceSpec(f"frontend/{fname}", RAW_BYTES_V1, p.stat().st_size))

    # Global LLM tensors
    specs.extend(
        [
            TensorSpec("text/embed", (201024, 2304), "BF16", LAYOUT),
            TensorSpec("text/embed_norm", (2304,), "BF16", LAYOUT),
            TensorSpec("text/final_norm", (2304,), "BF16", LAYOUT),
            TensorSpec("text/output_head", (201024, 2304), "BF16", LAYOUT),
            TensorSpec("text/attn_res/output_query", (2304,), "FP32", LAYOUT),
            TensorSpec("text/loop/cross_pass_queries", (24, 2304), "FP32", LAYOUT),
            TensorSpec("text/loop/cross_pass_kv_queries", (24, 2304), "FP32", LAYOUT),
        ]
    )

    # Per physical layer 0..23
    for ip in range(24):
        lp = f"text/layers/{ip}/"
        is_local = ip in LOCAL_LAYERS
        if ip > 0:
            q_shape = (2304,) if is_local else (3, 2304)
            specs.append(TensorSpec(lp + "attn_res/attn_query", q_shape, "FP32", LAYOUT))
        specs.append(TensorSpec(lp + "attn_res/mlp_query", (2304,), "FP32", LAYOUT))

        if is_local:
            specs.append(TensorSpec(lp + "attn_norm", (2304,), "BF16", LAYOUT))
        else:
            specs.append(TensorSpec(lp + "q_input_norm", (2304,), "BF16", LAYOUT))
            specs.append(TensorSpec(lp + "k_input_norm", (2304,), "BF16", LAYOUT))
            specs.append(TensorSpec(lp + "v_input_norm", (2304,), "BF16", LAYOUT))
        specs.append(TensorSpec(lp + "mlp_norm", (2304,), "BF16", LAYOUT))

        # Attention projections: fused [Q(3072), gate(3072), R(384), sink(24)] = 6552 rows
        specs.append(TensorSpec(lp + "attn/q_gate_r_sink", (6552, 2304), "BF16", LAYOUT))
        specs.append(TensorSpec(lp + "attn/k", (1024, 2304), "BF16", LAYOUT))
        specs.append(TensorSpec(lp + "attn/v", (1024, 2304), "BF16", LAYOUT))
        specs.append(TensorSpec(lp + "attn/o", (2304, 3072), "BF16", LAYOUT))
        specs.append(TensorSpec(lp + "attn/q_clip_scale", (24, 128), "BF16", LAYOUT))
        specs.append(TensorSpec(lp + "attn/rel_clip_scale", (24, 16), "BF16", LAYOUT))
        specs.append(TensorSpec(lp + "attn/rel_logits_proj", (16, 1024), "BF16", LAYOUT))

        if is_local:
            specs.append(TensorSpec(lp + "attn/k_sconv", (1024, 4), "BF16", LAYOUT))
            specs.append(TensorSpec(lp + "attn/v_sconv", (1024, 4), "BF16", LAYOUT))
        else:
            specs.append(TensorSpec(lp + "attn/k_sconv_static", (1024, 4), "BF16", LAYOUT))
            specs.append(TensorSpec(lp + "attn/v_sconv_static", (1024, 4), "BF16", LAYOUT))
            specs.append(TensorSpec(lp + "attn/k_sconv_proj", (32, 2304), "BF16", LAYOUT))
            specs.append(TensorSpec(lp + "attn/v_sconv_proj", (32, 2304), "BF16", LAYOUT))

        specs.append(TensorSpec(lp + "attn_sconv", (2304, 4), "BF16", LAYOUT))
        specs.append(TensorSpec(lp + "mlp_sconv", (2304, 4), "BF16", LAYOUT))

        if ip in CORE_LAYERS:
            for c in ("attn", "k", "v", "mlp"):
                specs.append(TensorSpec(lp + f"controllers/{c}/down", (64, 4608), "BF16", LAYOUT))
                specs.append(TensorSpec(lp + f"controllers/{c}/gain_forget", (4608, 64), "BF16", LAYOUT))

        # MoE router + experts (2 contiguous TP2 slabs along intermediate_size=1152 -> 576/slab)
        specs.append(TensorSpec(lp + "mlp/router_weight", (128, 2304), "BF16", LAYOUT))
        specs.append(TensorSpec(lp + "mlp/router_bias", (128,), "FP32", LAYOUT))
        if ip in NVFP4_LAYERS:
            specs.append(TensorSpec(lp + "mlp/experts/w13_codes", (2, 128, 1152, 1152), "U8", LAYOUT))
            specs.append(TensorSpec(lp + "mlp/experts/w13_scales", (2, 128, 1152, 144), "FP8_E4M3FN", LAYOUT))
            specs.append(TensorSpec(lp + "mlp/experts/w13_amax", (128,), "FP32", LAYOUT))
            specs.append(TensorSpec(lp + "mlp/experts/w13_denom", (1,), "FP32", LAYOUT))
            specs.append(TensorSpec(lp + "mlp/experts/w2_codes", (2, 128, 2304, 288), "U8", LAYOUT))
            specs.append(TensorSpec(lp + "mlp/experts/w2_scales", (2, 128, 2304, 36), "FP8_E4M3FN", LAYOUT))
            specs.append(TensorSpec(lp + "mlp/experts/w2_amax", (128,), "FP32", LAYOUT))
            specs.append(TensorSpec(lp + "mlp/experts/w2_denom", (1,), "FP32", LAYOUT))
        else:
            specs.append(TensorSpec(lp + "mlp/experts/w13_codes", (2, 128, 1152, 2304), "FP8_E4M3FN", LAYOUT))
            specs.append(TensorSpec(lp + "mlp/experts/w13_scales", (2, 128, 1152, 72), "U8", LAYOUT))
            specs.append(TensorSpec(lp + "mlp/experts/w2_codes", (2, 128, 2304, 576), "FP8_E4M3FN", LAYOUT))
            specs.append(TensorSpec(lp + "mlp/experts/w2_scales", (2, 128, 2304, 18), "U8", LAYOUT))

    if include_mtp:
        specs.extend(
            [
                TensorSpec("mtp/fuse_weight", (4, 4), "BF16", LAYOUT),
                TensorSpec("mtp/mask_embed", (2304,), "BF16", LAYOUT),
                TensorSpec("mtp/norm", (2304,), "BF16", LAYOUT),
            ]
        )
        for li in range(3):
            mp = f"mtp/layers/{li}/"
            specs.extend(
                [
                    TensorSpec(mp + "dflare_norm", (2304,), "BF16", LAYOUT),
                    TensorSpec(mp + "input_norm", (2304,), "BF16", LAYOUT),
                    TensorSpec(mp + "post_attention_norm", (2304,), "BF16", LAYOUT),
                    TensorSpec(mp + "attention/qkv", (5120, 2304), "BF16", LAYOUT),
                    TensorSpec(mp + "attention/kv_ctx", (2048, 2304), "BF16", LAYOUT),
                    TensorSpec(mp + "attention/q_norm", (128,), "BF16", LAYOUT),
                    TensorSpec(mp + "attention/k_norm", (128,), "BF16", LAYOUT),
                    TensorSpec(mp + "attention/k_norm_ctx", (128,), "BF16", LAYOUT),
                    TensorSpec(mp + "attention/o", (2304, 3072), "BF16", LAYOUT),
                    TensorSpec(mp + "mlp/gate_up", (18432, 2304), "BF16", LAYOUT),
                    TensorSpec(mp + "mlp/down", (2304, 9216), "BF16", LAYOUT),
                ]
            )
        specs.extend(
            [
                TensorSpec("mtp/markov/dflare_norm", (2304,), "BF16", LAYOUT),
                TensorSpec("mtp/markov/w1", (201024, 256), "BF16", LAYOUT),
                TensorSpec("mtp/markov/rnn_proj_weight", (768, 5120), "BF16", LAYOUT),
                TensorSpec("mtp/markov/rnn_proj_bias", (768,), "BF16", LAYOUT),
                TensorSpec("mtp/markov/hidden_out_weight", (2304, 256), "BF16", LAYOUT),
                TensorSpec("mtp/markov/hidden_out_bias", (2304,), "BF16", LAYOUT),
            ]
        )

    return specs


def convert(model_dir: Path, output_path: Path, include_mtp: bool) -> None:
    t0 = time.time()
    _validate_config(model_dir)
    mtp_path = model_dir / "mtp.safetensors"
    if include_mtp and not mtp_path.is_file():
        raise FileNotFoundError(f"MTP requested but missing: {mtp_path}")

    specs = build_inventory(model_dir, include_mtp)
    identity = ArtifactIdentity(model_id="inkling2-mini", weights_id="nvfp4-mxfp8")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    main_sf = safe_open(str(model_dir / "model-00001-of-00001.safetensors"), framework="pt", device="cpu")
    mtp_sf = safe_open(str(mtp_path), framework="pt", device="cpu") if include_mtp else None

    with ArtifactWriter(output_path, identity, specs) as writer:
        # 1. Frontend resources
        for fname in (
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "generation_config.json",
        ):
            writer.write(f"frontend/{fname}", (model_dir / fname).read_bytes())

        # 2. Global LLM tensors
        writer.write("text/embed", _tensor_bytes(_require_tensor(main_sf, "model.llm.embed.weight", torch.bfloat16, (201024, 2304))))
        writer.write("text/embed_norm", _tensor_bytes(_require_tensor(main_sf, "model.llm.embed_norm.weight", torch.bfloat16, (2304,))))
        writer.write("text/final_norm", _tensor_bytes(_require_tensor(main_sf, "model.llm.norm.weight", torch.bfloat16, (2304,))))
        writer.write("text/output_head", _tensor_bytes(_require_tensor(main_sf, "model.llm.unembed.weight", torch.bfloat16, (201024, 2304))))
        writer.write("text/attn_res/output_query", _tensor_bytes(_require_tensor(main_sf, "model.llm.attn_res.output_query", torch.float32, (2304,))))
        writer.write("text/loop/cross_pass_queries", _tensor_bytes(_require_tensor(main_sf, "model.llm.middle_cycle_cross_pass_queries", torch.float32, (24, 2304))))
        writer.write("text/loop/cross_pass_kv_queries", _tensor_bytes(_require_tensor(main_sf, "model.llm.middle_cycle_cross_pass_kv_queries", torch.float32, (24, 2304))))

        # 3. Per-layer tensors
        for ip in range(24):
            lp = f"text/layers/{ip}/"
            sp = f"model.llm.layers.{ip}."
            is_local = ip in LOCAL_LAYERS

            if ip > 0:
                q_shape = (2304,) if is_local else (3, 2304)
                writer.write(lp + "attn_res/attn_query", _tensor_bytes(_require_tensor(main_sf, f"model.llm.attn_res.attn_queries.{ip - 1}", torch.float32, q_shape)))
            writer.write(lp + "attn_res/mlp_query", _tensor_bytes(_require_tensor(main_sf, f"model.llm.attn_res.mlp_queries.{ip}", torch.float32, (2304,))))

            if is_local:
                writer.write(lp + "attn_norm", _tensor_bytes(_require_tensor(main_sf, sp + "attn_norm.weight", torch.bfloat16, (2304,))))
            else:
                writer.write(lp + "q_input_norm", _tensor_bytes(_require_tensor(main_sf, sp + "q_input_norm.weight", torch.bfloat16, (2304,))))
                writer.write(lp + "k_input_norm", _tensor_bytes(_require_tensor(main_sf, sp + "k_input_norm.weight", torch.bfloat16, (2304,))))
                writer.write(lp + "v_input_norm", _tensor_bytes(_require_tensor(main_sf, sp + "v_input_norm.weight", torch.bfloat16, (2304,))))
            writer.write(lp + "mlp_norm", _tensor_bytes(_require_tensor(main_sf, sp + "mlp_norm.weight", torch.bfloat16, (2304,))))

            wq = _require_tensor(main_sf, sp + "attn.wq_du.weight", torch.bfloat16, (3072, 2304))
            wg = _require_tensor(main_sf, sp + "attn.gate_attn.weight", torch.bfloat16, (3072, 2304))
            wr = _require_tensor(main_sf, sp + "attn.wr_du.weight", torch.bfloat16, (384, 2304))
            ws = _require_tensor(main_sf, sp + "attn.attn_sink_proj.weight", torch.bfloat16, (24, 2304))
            writer.write(lp + "attn/q_gate_r_sink", _tensor_bytes(torch.cat([wq, wg, wr, ws], dim=0)))

            writer.write(lp + "attn/k", _tensor_bytes(_require_tensor(main_sf, sp + "attn.wk_dv.weight", torch.bfloat16, (1024, 2304))))
            writer.write(lp + "attn/v", _tensor_bytes(_require_tensor(main_sf, sp + "attn.wv_dv.weight", torch.bfloat16, (1024, 2304))))
            writer.write(lp + "attn/o", _tensor_bytes(_require_tensor(main_sf, sp + "attn.wo_ud.weight", torch.bfloat16, (2304, 3072))))
            writer.write(lp + "attn/q_clip_scale", _tensor_bytes(_require_tensor(main_sf, sp + "attn.q_l2_clip.scale", torch.bfloat16, (24, 128))))
            writer.write(lp + "attn/rel_clip_scale", _tensor_bytes(_require_tensor(main_sf, sp + "attn.rel_l2_clip.scale", torch.bfloat16, (24, 16))))
            writer.write(lp + "attn/rel_logits_proj", _tensor_bytes(_require_tensor(main_sf, sp + "attn.rel_logits_proj.proj", torch.bfloat16, (16, 1024))))

            if is_local:
                ks = _require_tensor(main_sf, sp + "attn.k_sconv.weight", torch.bfloat16, (1024, 1, 4)).squeeze(1)
                vs = _require_tensor(main_sf, sp + "attn.v_sconv.weight", torch.bfloat16, (1024, 1, 4)).squeeze(1)
                writer.write(lp + "attn/k_sconv", _tensor_bytes(ks))
                writer.write(lp + "attn/v_sconv", _tensor_bytes(vs))
            else:
                ks = _require_tensor(main_sf, sp + "attn.k_sconv.static_w", torch.bfloat16, (4, 1024)).T.contiguous()
                vs = _require_tensor(main_sf, sp + "attn.v_sconv.static_w", torch.bfloat16, (4, 1024)).T.contiguous()
                kp = _require_tensor(main_sf, sp + "attn.k_sconv.weight_proj", torch.bfloat16, (32, 2304))
                vp = _require_tensor(main_sf, sp + "attn.v_sconv.weight_proj", torch.bfloat16, (32, 2304))
                writer.write(lp + "attn/k_sconv_static", _tensor_bytes(ks))
                writer.write(lp + "attn/v_sconv_static", _tensor_bytes(vs))
                writer.write(lp + "attn/k_sconv_proj", _tensor_bytes(kp))
                writer.write(lp + "attn/v_sconv_proj", _tensor_bytes(vp))

            asc = _require_tensor(main_sf, sp + "attn_sconv.weight", torch.bfloat16, (2304, 1, 4)).squeeze(1)
            msc = _require_tensor(main_sf, sp + "mlp_sconv.weight", torch.bfloat16, (2304, 1, 4)).squeeze(1)
            writer.write(lp + "attn_sconv", _tensor_bytes(asc))
            writer.write(lp + "mlp_sconv", _tensor_bytes(msc))

            if ip in CORE_LAYERS:
                for c in ("attn", "k", "v", "mlp"):
                    cp = sp + f"controllers.{c}."
                    dw = _require_tensor(main_sf, cp + "down_weight", torch.bfloat16, (64, 4608))
                    gw = _require_tensor(main_sf, cp + "gain_weight", torch.bfloat16, (2304, 64))
                    fw = _require_tensor(main_sf, cp + "forget_weight", torch.bfloat16, (2304, 64))
                    writer.write(lp + f"controllers/{c}/down", _tensor_bytes(dw))
                    writer.write(lp + f"controllers/{c}/gain_forget", _tensor_bytes(torch.cat([gw, fw], dim=0)))

            writer.write(lp + "mlp/router_weight", _tensor_bytes(_require_tensor(main_sf, sp + "mlp.gate.weight", torch.bfloat16, (128, 2304))))
            writer.write(lp + "mlp/router_bias", _tensor_bytes(_require_tensor(main_sf, sp + "mlp.gate.bias", torch.float32, (128,))))

            if ip in NVFP4_LAYERS:
                w13 = _require_tensor(main_sf, sp + "mlp.experts.w13_weight", torch.uint8, (128, 2304, 1152))
                w13s = _require_tensor(main_sf, sp + "mlp.experts.w13_weight_scale", torch.float8_e4m3fn, (128, 2304, 144))
                w13a = _require_tensor(main_sf, sp + "mlp.experts.w13_weight_amax", torch.float32, (128,))
                w13d = _require_tensor(main_sf, sp + "mlp.experts.w13_weight_outer_scale_denominator", torch.float32, (1,))
                w2 = _require_tensor(main_sf, sp + "mlp.experts.w2_weight", torch.uint8, (128, 2304, 576))
                w2s = _require_tensor(main_sf, sp + "mlp.experts.w2_weight_scale", torch.float8_e4m3fn, (128, 2304, 72))
                w2a = _require_tensor(main_sf, sp + "mlp.experts.w2_weight_amax", torch.float32, (128,))
                w2d = _require_tensor(main_sf, sp + "mlp.experts.w2_weight_outer_scale_denominator", torch.float32, (1,))

                # Split into 2 contiguous TP2 intermediate slabs along axis 1 (w13) and axis 2 (w2)
                w13_slabs = torch.stack([w13[:, :1152, :], w13[:, 1152:, :]], dim=0)
                w13s_slabs = torch.stack([w13s[:, :1152, :], w13s[:, 1152:, :]], dim=0)
                w2_slabs = torch.stack([w2[:, :, :288], w2[:, :, 288:]], dim=0)
                w2s_slabs = torch.stack([w2s[:, :, :36], w2s[:, :, 36:]], dim=0)

                writer.write(lp + "mlp/experts/w13_codes", _tensor_bytes(w13_slabs))
                writer.write(lp + "mlp/experts/w13_scales", _tensor_bytes(w13s_slabs))
                writer.write(lp + "mlp/experts/w13_amax", _tensor_bytes(w13a))
                writer.write(lp + "mlp/experts/w13_denom", _tensor_bytes(w13d))
                writer.write(lp + "mlp/experts/w2_codes", _tensor_bytes(w2_slabs))
                writer.write(lp + "mlp/experts/w2_scales", _tensor_bytes(w2s_slabs))
                writer.write(lp + "mlp/experts/w2_amax", _tensor_bytes(w2a))
                writer.write(lp + "mlp/experts/w2_denom", _tensor_bytes(w2d))
            else:
                w13 = _require_tensor(main_sf, sp + "mlp.experts.w13_weight", torch.float8_e4m3fn, (128, 2304, 2304))
                w13s = _require_tensor(main_sf, sp + "mlp.experts.w13_weight_scale", torch.uint8, (128, 2304, 72))
                w2 = _require_tensor(main_sf, sp + "mlp.experts.w2_weight", torch.float8_e4m3fn, (128, 2304, 1152))
                w2s = _require_tensor(main_sf, sp + "mlp.experts.w2_weight_scale", torch.uint8, (128, 2304, 36))

                w13_slabs = torch.stack([w13[:, :1152, :], w13[:, 1152:, :]], dim=0)
                w13s_slabs = torch.stack([w13s[:, :1152, :], w13s[:, 1152:, :]], dim=0)
                w2_slabs = torch.stack([w2[:, :, :576], w2[:, :, 576:]], dim=0)
                w2s_slabs = torch.stack([w2s[:, :, :18], w2s[:, :, 18:]], dim=0)

                writer.write(lp + "mlp/experts/w13_codes", _tensor_bytes(w13_slabs))
                writer.write(lp + "mlp/experts/w13_scales", _tensor_bytes(w13s_slabs))
                writer.write(lp + "mlp/experts/w2_codes", _tensor_bytes(w2_slabs))
                writer.write(lp + "mlp/experts/w2_scales", _tensor_bytes(w2s_slabs))

            print(f"  converted layer {ip:2d}/23 ({time.time() - t0:.1f}s)", flush=True)

        # 4. Optional DSpark MTP package
        if include_mtp and mtp_sf is not None:
            writer.write("mtp/fuse_weight", _tensor_bytes(_require_tensor(mtp_sf, "fuse_weight", torch.bfloat16, (4, 4))))
            writer.write("mtp/mask_embed", _tensor_bytes(_require_tensor(mtp_sf, "mask_embed", torch.bfloat16, (2304,))))
            writer.write("mtp/norm", _tensor_bytes(_require_tensor(mtp_sf, "norm.weight", torch.bfloat16, (2304,))))
            for li in range(3):
                mp = f"mtp/layers/{li}/"
                sp = f"layers.{li}."
                writer.write(mp + "dflare_norm", _tensor_bytes(_require_tensor(mtp_sf, sp + "dflare_norm.weight", torch.bfloat16, (2304,))))
                writer.write(mp + "input_norm", _tensor_bytes(_require_tensor(mtp_sf, sp + "input_layernorm.weight", torch.bfloat16, (2304,))))
                writer.write(mp + "post_attention_norm", _tensor_bytes(_require_tensor(mtp_sf, sp + "post_attention_layernorm.weight", torch.bfloat16, (2304,))))

                q = _require_tensor(mtp_sf, sp + "self_attn.q_proj.weight", torch.bfloat16, (3072, 2304))
                k = _require_tensor(mtp_sf, sp + "self_attn.k_proj.weight", torch.bfloat16, (1024, 2304))
                v = _require_tensor(mtp_sf, sp + "self_attn.v_proj.weight", torch.bfloat16, (1024, 2304))
                writer.write(mp + "attention/qkv", _tensor_bytes(torch.cat([q, k, v], dim=0)))

                kc = _require_tensor(mtp_sf, sp + "self_attn.k_proj_ctx.weight", torch.bfloat16, (1024, 2304))
                vc = _require_tensor(mtp_sf, sp + "self_attn.v_proj_ctx.weight", torch.bfloat16, (1024, 2304))
                writer.write(mp + "attention/kv_ctx", _tensor_bytes(torch.cat([kc, vc], dim=0)))

                writer.write(mp + "attention/q_norm", _tensor_bytes(_require_tensor(mtp_sf, sp + "self_attn.q_norm.weight", torch.bfloat16, (128,))))
                writer.write(mp + "attention/k_norm", _tensor_bytes(_require_tensor(mtp_sf, sp + "self_attn.k_norm.weight", torch.bfloat16, (128,))))
                writer.write(mp + "attention/k_norm_ctx", _tensor_bytes(_require_tensor(mtp_sf, sp + "self_attn.k_norm_ctx.weight", torch.bfloat16, (128,))))
                writer.write(mp + "attention/o", _tensor_bytes(_require_tensor(mtp_sf, sp + "self_attn.o_proj.weight", torch.bfloat16, (2304, 3072))))

                gp = _require_tensor(mtp_sf, sp + "mlp.gate_proj.weight", torch.bfloat16, (9216, 2304))
                up = _require_tensor(mtp_sf, sp + "mlp.up_proj.weight", torch.bfloat16, (9216, 2304))
                writer.write(mp + "mlp/gate_up", _tensor_bytes(torch.cat([gp, up], dim=0)))
                writer.write(mp + "mlp/down", _tensor_bytes(_require_tensor(mtp_sf, sp + "mlp.down_proj.weight", torch.bfloat16, (2304, 9216))))

            writer.write("mtp/markov/dflare_norm", _tensor_bytes(_require_tensor(mtp_sf, "markov_head.dflare_norm.weight", torch.bfloat16, (2304,))))
            writer.write("mtp/markov/w1", _tensor_bytes(_require_tensor(mtp_sf, "markov_head.markov_w1.weight", torch.bfloat16, (201024, 256))))
            writer.write("mtp/markov/rnn_proj_weight", _tensor_bytes(_require_tensor(mtp_sf, "markov_head.rnn_proj.weight", torch.bfloat16, (768, 5120))))
            writer.write("mtp/markov/rnn_proj_bias", _tensor_bytes(_require_tensor(mtp_sf, "markov_head.rnn_proj.bias", torch.bfloat16, (768,))))
            writer.write("mtp/markov/hidden_out_weight", _tensor_bytes(_require_tensor(mtp_sf, "markov_head.hidden_out.weight", torch.bfloat16, (2304, 256))))
            writer.write("mtp/markov/hidden_out_bias", _tensor_bytes(_require_tensor(mtp_sf, "markov_head.hidden_out.bias", torch.bfloat16, (2304,))))
            print(f"  converted DSpark MTP package ({time.time() - t0:.1f}s)", flush=True)

    size_gb = output_path.stat().st_size / (1024**3)
    print(f"Wrote {output_path} ({size_gb:.3f} GiB, {len(specs)} objects) in {time.time() - t0:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--no-mtp", action="store_true")
    args = ap.parse_args()
    convert(args.model_dir, args.output, include_mtp=not args.no_mtp)


if __name__ == "__main__":
    main()
