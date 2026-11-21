#!/usr/bin/env python3
"""Independent PyTorch reference oracle for thinkingmachines/Inkling-2-Mini + DSpark MTP.

Preserves exact checkpoint NVFP4 (layers 0..19) and MXFP8 (layers 20..23) codes/scales
and executes the full 48-logical-layer looped architecture (prelude 0..5, 3 passes over 6..17,
epilogue 18..23) plus optional DSpark MTP drafter (`mtp.safetensors`).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import AutoTokenizer


def _build_e4m3_table(device: torch.device) -> torch.Tensor:
    codes = np.arange(256, dtype=np.uint8)
    exponent = ((codes & 0x7F) >> 3).astype(np.int32)
    mantissa = (codes & 7).astype(np.float32)
    values = np.where(
        exponent == 0,
        np.ldexp(mantissa, -9),
        np.ldexp(1.0 + mantissa / 8.0, exponent - 7),
    )
    values = np.where(codes & 0x80, -values, values)
    values[(codes & 0x7F) == 0x7F] = np.nan
    return torch.from_numpy(values.astype(np.float32)).to(device)


def _build_e2m1_table(device: torch.device) -> torch.Tensor:
    arr = np.array(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        dtype=np.float32,
    )
    return torch.from_numpy(arr).to(device)


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    return y


def l2_clip(x: torch.Tensor, scale: torch.Tensor | None, threshold: float, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    if scale is not None:
        xf = xf * scale.float()
    norm = torch.linalg.vector_norm(xf, dim=-1, keepdim=True)
    factor = (threshold / (norm + eps)).clamp(max=1.0)
    return xf * factor


def block_attn_res(sources: list[torch.Tensor], queries: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # sources: list of [T, H], queries: [Q, H] or [H]
    if queries.ndim == 1:
        queries = queries.unsqueeze(0)
    stacked = torch.stack([s.float() for s in sources], dim=-2)  # [T, S, H]
    normed = rms_norm(stacked, None, eps)  # [T, S, H]
    q = queries.float()  # [Q, H]
    scale = queries.shape[-1] ** -0.5
    scores = torch.einsum("qh,tsh->tqs", q, normed) * scale  # [T, Q, S]
    probs = F.softmax(scores, dim=-1)
    out = torch.einsum("tqs,tsh->tqh", probs, normed)  # [T, Q, H]
    return out


class StaticSConvState:
    def __init__(self, channels: int, k: int, device: torch.device):
        self.buf = torch.zeros(k - 1, channels, dtype=torch.float32, device=device)
        self.k = k

    def step(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # x: [T, C], weight: [C, 1, K] (weight[:, 0, K-1] is lag 0 / current token)
        T, C = x.shape
        xf = x.float()
        w = weight.float().squeeze(1)  # [C, K]
        full = torch.cat([self.buf, xf], dim=0)  # [T + K - 1, C]
        out = torch.zeros_like(xf)
        for tap in range(self.k):
            # tap 0 -> oldest (lag K-1), tap K-1 -> current (lag 0)
            out = out + full[tap : tap + T] * w[:, tap].unsqueeze(0)
        self.buf = full[-(self.k - 1) :].clone()
        return (out + xf) / math.sqrt(self.k + 1)


class DynamicSConvState:
    def __init__(self, channels: int, k: int, device: torch.device):
        self.buf = torch.zeros(k - 1, channels, dtype=torch.float32, device=device)
        self.k = k

    def step(
        self,
        x: torch.Tensor,
        dyn_in: torch.Tensor,
        static_w: torch.Tensor,
        weight_proj: torch.Tensor,
        num_heads: int,
    ) -> torch.Tensor:
        # x: [T, C], dyn_in: [T, H], static_w: [K, C] (static_w[lag] is lag steps back),
        # weight_proj: [num_heads * K, H]
        T, C = x.shape
        head_dim = C // num_heads
        xf = x.float()
        dyn = F.linear(dyn_in.float(), weight_proj.float()).view(T, num_heads, 1, self.k)
        sw = static_w.float().T.view(1, num_heads, head_dim, self.k)  # [1, Hkv, D, K]
        w = sw + dyn  # [T, Hkv, D, K], index lag = 0..K-1
        full = torch.cat([self.buf, xf], dim=0)  # [T + K - 1, C]
        out = torch.zeros_like(xf)
        for lag in range(self.k):
            start = (self.k - 1) - lag
            slice_x = full[start : start + T].view(T, num_heads, head_dim)
            out = out + (slice_x * w[:, :, :, lag]).reshape(T, C)
        self.buf = full[-(self.k - 1) :].clone()
        return (out + xf) / math.sqrt(2 * self.k + 1)


class Inkling2MiniReference:
    def __init__(self, model_dir: Path, device: torch.device, bf16_boundaries: bool = True):
        self.model_dir = model_dir
        self.device = device
        self.bf16_boundaries = bf16_boundaries
        self.config = json.loads((model_dir / "config.json").read_text())
        self.tc = self.config["text_config"]
        self.hidden = self.tc["hidden_size"]  # 2304
        self.n_phys = self.tc["num_hidden_layers"]  # 24
        self.n_heads = self.tc["num_attention_heads"]  # 24
        self.n_kv_heads = self.tc["num_key_value_heads"]  # 8
        self.head_dim = self.tc["head_dim"]  # 128
        self.d_rel = self.tc["d_rel"]  # 16
        self.rel_extent = self.tc["rel_extent"]  # 1024
        self.swa = self.tc["sliding_window_size"]  # 1024
        self.local_layers = set(self.tc["local_layer_ids"])
        self.n_block = self.tc["n_layers_per_attn_res_block"]  # 6
        self.top_k = self.tc["num_experts_per_tok"]  # 8
        self.route_scale = float(self.tc["route_scale"])  # 8.0
        self.gate_cap = float(self.tc["mlp_gate_softcap"])  # 4.0
        self.up_cap = float(self.tc["mlp_up_softcap"])  # 25.0
        self.qk_clip = float(self.tc["qk_l2_clip_threshold"])
        self.rel_clip = float(self.tc["rel_l2_clip_threshold"])
        self.log_floor = float(self.tc["log_scaling_n_floor"])
        self.log_alpha = float(self.tc["log_scaling_alpha"])
        self.eps = float(self.tc["rms_norm_eps"])
        self.softcap = float(self.tc["final_logit_softcapping"])
        self.unpadded_vocab = int(self.tc["unpadded_vocab_size"])
        self.loop = self.tc["loop_config"]
        self.core_start = int(self.loop["core_start"])  # 6
        self.core_end = int(self.loop["core_end"])  # 18
        self.num_passes = int(self.loop["num_passes"])  # 3
        self.n_logical = self.n_phys + (self.num_passes - 1) * (self.core_end - self.core_start)  # 48

        self.e4m3_lut = _build_e4m3_table(device)
        self.e2m1_lut = _build_e2m1_table(device)

        t0 = time.time()
        self.sf = safe_open(str(model_dir / "model-00001-of-00001.safetensors"), framework="pt", device="cpu")
        self.weights: dict[str, torch.Tensor] = {}
        self.expert_cpu: dict[str, torch.Tensor] = {}
        # Load non-expert LLM tensors onto GPU; keep packed 3D expert weights in CPU RAM (only 8/128 fetched per step!)
        for k in self.sf.keys():
            if not k.startswith("model.llm."):
                continue
            short = k[len("model.llm.") :]
            t = self.sf.get_tensor(k)
            if ".mlp.experts.w" in short and (short.endswith("_weight") or short.endswith("_weight_scale")):
                self.expert_cpu[short] = t.contiguous()
            else:
                self.weights[short] = t.to(device)
        print(f"Loaded LLM weights in {time.time() - t0:.2f}s (GPU non-expert: {len(self.weights)}, CPU packed experts: {len(self.expert_cpu)})")

        # Load MTP weights if present
        mtp_path = model_dir / "mtp.safetensors"
        self.mtp: dict[str, torch.Tensor] = {}
        if mtp_path.is_file():
            with safe_open(str(mtp_path), framework="pt", device="cpu") as msf:
                for k in msf.keys():
                    self.mtp[k] = msf.get_tensor(k).to(device)
            print(f"Loaded MTP weights: {len(self.mtp)} tensors")

    def _bound(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(torch.bfloat16).float() if self.bf16_boundaries else x.float()

    def _lin(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return F.linear(self._bound(x), w.float())

    def _dequant_expert(self, ip: int, kind: str, eid: int) -> torch.Tensor:
        # kind: "w13" or "w2"
        prefix = f"layers.{ip}.mlp.experts.{kind}_weight"
        packed = self.expert_cpu[prefix][eid].to(self.device)
        scale = self.expert_cpu[prefix + "_scale"][eid].to(self.device)
        if ip < 20:
            # NVFP4: packed U8 [N, K/2], scale F8_E4M3 [N, K/16]
            p_u8 = packed.view(torch.uint8).long()
            s_u8 = scale.view(torch.uint8).long()
            lo = self.e2m1_lut[p_u8 & 0xF]
            hi = self.e2m1_lut[(p_u8 >> 4) & 0xF]
            vals = torch.stack([lo, hi], dim=-1).reshape(p_u8.shape[0], p_u8.shape[1] * 2)
            s_f = self.e4m3_lut[s_u8]
            vals = (vals.view(vals.shape[0], -1, 16) * s_f.unsqueeze(-1)).reshape(vals.shape)
            amax = self.weights[prefix + "_amax"][eid].float()
            denom = self.weights[prefix + "_outer_scale_denominator"][0].float()
            return vals * (amax / denom)
        else:
            # MXFP8: packed F8_E4M3 [N, K], scale U8 [N, K/32] (UE8M0)
            p_u8 = packed.view(torch.uint8).long()
            s_u8 = scale.view(torch.uint8).to(torch.int32)
            vals = self.e4m3_lut[p_u8]
            # UE8M0: 2^(s - 127)
            s_f = torch.ldexp(torch.ones_like(s_u8, dtype=torch.float32), s_u8 - 127)
            return (vals.view(vals.shape[0], -1, 32) * s_f.unsqueeze(-1)).reshape(vals.shape)

    def _control(self, cur: torch.Tensor, msg: torch.Tensor, ip: int, kind: str) -> torch.Tensor:
        p = f"layers.{ip}.controllers.{kind}."
        z = torch.cat([rms_norm(cur, None, self.eps), rms_norm(msg, None, self.eps)], dim=-1)
        h = self._lin(z, self.weights[p + "down_weight"])
        g = torch.tanh(self._lin(h, self.weights[p + "gain_weight"]))
        f = torch.tanh(self._lin(h, self.weights[p + "forget_weight"]))
        return (1.0 - f) * cur.float() + g * msg.float()

    def init_cache(self) -> dict[str, Any]:
        k_dim = self.n_kv_heads * self.head_dim
        sconv = []
        for il in range(self.n_logical):
            ip = self._phys_of_logical(il)
            is_local = ip in self.local_layers
            k_sc = StaticSConvState(k_dim, 4, self.device) if is_local else DynamicSConvState(k_dim, 4, self.device)
            v_sc = StaticSConvState(k_dim, 4, self.device) if is_local else DynamicSConvState(k_dim, 4, self.device)
            a_sc = StaticSConvState(self.hidden, 4, self.device)
            m_sc = StaticSConvState(self.hidden, 4, self.device)
            sconv.append((k_sc, v_sc, a_sc, m_sc))
        return {
            "seq_len": 0,
            "k": [None] * self.n_logical,
            "v": [None] * self.n_logical,
            "sconv": sconv,
        }

    def _phys_of_logical(self, il: int) -> int:
        # Chronological logical index:
        # 0..5 -> 0..5
        # 6..17 -> pass 0 (6..17)
        # 18..29 -> pass 1 (6..17)
        # 30..41 -> pass 2 (6..17)
        # 42..47 -> 18..23
        if il < self.core_start:
            return il
        core_len = self.core_end - self.core_start
        if il < self.core_start + self.num_passes * core_len:
            return self.core_start + (il - self.core_start) % core_len
        return self.core_end + (il - (self.core_start + self.num_passes * core_len))

    def _run_layer(
        self,
        il: int,
        ip: int,
        cur: torch.Tensor,
        blocks: list[torch.Tensor],
        cache: dict[str, Any],
        positions: torch.Tensor,
        messages: torch.Tensor | None = None,
    ) -> torch.Tensor:
        T = cur.shape[0]
        is_local = ip in self.local_layers
        k_sc, v_sc, a_sc, m_sc = cache["sconv"][il]
        lp = f"layers.{ip}."

        inpSA = cur
        if ip > 0:
            mixed = block_attn_res([*blocks, cur], self.weights[f"attn_res.attn_queries.{ip - 1}"], self.eps)
            if is_local:
                q_in = k_in = v_in = mixed[:, 0, :]
            else:
                q_in, k_in, v_in = mixed[:, 0, :], mixed[:, 1, :], mixed[:, 2, :]
        else:
            q_in = k_in = v_in = cur

        if messages is not None:
            q_in = self._control(q_in, messages[:, 0, :], ip, "attn")
            k_in = self._control(k_in, messages[:, 1, :], ip, "k")
            v_in = self._control(v_in, messages[:, 2, :], ip, "v")

        if is_local:
            w_ln = self.weights[lp + "attn_norm.weight"]
            if messages is not None:
                q_in = self._bound(rms_norm(q_in, w_ln, self.eps))
                k_in = self._bound(rms_norm(k_in, w_ln, self.eps))
                v_in = self._bound(rms_norm(v_in, w_ln, self.eps))
            else:
                q_in = self._bound(rms_norm(q_in, w_ln, self.eps))
                k_in = v_in = q_in
        else:
            q_in = self._bound(rms_norm(q_in, self.weights[lp + "q_input_norm.weight"], self.eps))
            k_in = self._bound(rms_norm(k_in, self.weights[lp + "k_input_norm.weight"], self.eps))
            v_in = self._bound(rms_norm(v_in, self.weights[lp + "v_input_norm.weight"], self.eps))

        ap = lp + "attn."
        Q = self._lin(q_in, self.weights[ap + "wq_du.weight"])
        K = self._lin(k_in, self.weights[ap + "wk_dv.weight"])
        V = self._lin(v_in, self.weights[ap + "wv_dv.weight"])
        R = self._lin(q_in, self.weights[ap + "wr_du.weight"])
        gate = self._lin(q_in, self.weights[ap + "gate_attn.weight"])
        sinks = self._lin(q_in, self.weights[ap + "attn_sink_proj.weight"])  # [T, Hq]

        if is_local:
            K = k_sc.step(K, self.weights[ap + "k_sconv.weight"])
            V = v_sc.step(V, self.weights[ap + "v_sconv.weight"])
        else:
            K = k_sc.step(K, k_in, self.weights[ap + "k_sconv.static_w"], self.weights[ap + "k_sconv.weight_proj"], self.n_kv_heads)
            V = v_sc.step(V, v_in, self.weights[ap + "v_sconv.static_w"], self.weights[ap + "v_sconv.weight_proj"], self.n_kv_heads)

        Q = l2_clip(Q.view(T, self.n_heads, self.head_dim), self.weights[ap + "q_l2_clip.scale"], self.qk_clip, self.eps)
        K = l2_clip(K.view(T, self.n_kv_heads, self.head_dim), None, self.qk_clip, self.eps)
        V = rms_norm(V.view(T, self.n_kv_heads, self.head_dim), None, self.eps)
        R = l2_clip(R.view(T, self.n_heads, self.d_rel), self.weights[ap + "rel_l2_clip.scale"], self.rel_clip, self.eps)

        if not is_local:
            tau = 1.0 + self.log_alpha * torch.log(((positions.float() + 1.0) / self.log_floor).clamp(min=1.0))
            tau = tau[:, None, None]
            Q = self._bound(Q) * tau
            R = self._bound(R) * tau

        Q = self._bound(Q)
        K = self._bound(K)
        V = self._bound(V)
        R = self._bound(R)

        # Update KV cache
        if cache["k"][il] is None:
            cache["k"][il] = K
            cache["v"][il] = V
        else:
            cache["k"][il] = torch.cat([cache["k"][il], K], dim=0)
            cache["v"][il] = torch.cat([cache["v"][il], V], dim=0)

        K_all = cache["k"][il]  # [S, Hkv, D]
        V_all = cache["v"][il]
        S = K_all.shape[0]
        kv_pos = torch.arange(S, device=self.device)

        # Relative bias: R [T, Hq, 16] @ proj [16, 1024] -> [T, Hq, 1024]
        rel_table = torch.matmul(R, self.weights[ap + "rel_logits_proj.proj"].float())  # [T, Hq, 1024]
        dist = positions[:, None] - kv_pos[None, :]  # [T, S]
        valid = dist >= 0
        if is_local:
            valid = valid & (dist < self.swa)
        in_rel = (dist >= 0) & (dist < self.rel_extent)
        gather_idx = dist.clamp(0, self.rel_extent - 1)[:, None, :].expand(T, self.n_heads, S)
        rel_bias = rel_table.gather(-1, gather_idx)
        rel_bias = rel_bias.masked_fill(~in_rel[:, None, :], 0.0)

        # GQA repeat
        rep = self.n_heads // self.n_kv_heads
        K_rep = K_all.repeat_interleave(rep, dim=1)  # [S, Hq, D]
        V_rep = V_all.repeat_interleave(rep, dim=1)
        scores = torch.einsum("thd,shd->ths", Q, K_rep) * (self.head_dim**-0.5) + rel_bias
        scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
        # Append sink logit per query/head
        comb = torch.cat([scores, sinks[:, :, None]], dim=-1)  # [T, Hq, S+1]
        probs = F.softmax(comb, dim=-1)[..., :-1]  # [T, Hq, S]
        attn_out = torch.einsum("ths,shd->thd", probs, V_rep).reshape(T, self.n_heads * self.head_dim)
        attn_out = self._bound(attn_out * torch.sigmoid(gate))
        attn_out = self._lin(attn_out, self.weights[ap + "wo_ud.weight"])
        attn_out = a_sc.step(attn_out, self.weights[lp + "attn_sconv.weight"])
        attn_out = self._bound(attn_out)

        if ip % self.n_block == 0:
            blocks.append(inpSA)
            ffn_inp = attn_out
        else:
            ffn_inp = inpSA.float() + attn_out.float()

        # MoE
        mlp_in = block_attn_res([*blocks, ffn_inp], self.weights[f"attn_res.mlp_queries.{ip}"], self.eps)[:, 0, :]
        if messages is not None:
            mlp_in = self._control(mlp_in, messages[:, 3, :], ip, "mlp")
        mlp_in = self._bound(rms_norm(mlp_in, self.weights[lp + "mlp_norm.weight"], self.eps))

        # Router (FP32)
        r_logits = F.linear(mlp_in.float(), self.weights[lp + "mlp.gate.weight"].float())
        r_scores = torch.sigmoid(r_logits)
        r_bias = self.weights[lp + "mlp.gate.bias"].float()
        topk_idx = torch.topk(r_scores + r_bias, self.top_k, dim=-1, sorted=False).indices  # [T, 8]
        topk_w = r_scores.gather(1, topk_idx)
        topk_w = (topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20)) * self.route_scale

        # Sparse experts
        moe_out = torch.zeros(T, self.hidden, dtype=torch.float32, device=self.device)
        unique_experts = topk_idx.unique().tolist()
        for eid in unique_experts:
            w13 = self._dequant_expert(ip, "w13", eid)  # [2304, 2304] interleaved (gate, up)
            w2 = self._dequant_expert(ip, "w2", eid)  # [2304, 1152]
            tok_mask = topk_idx == eid
            t_ids, k_ids = torch.where(tok_mask)
            x_e = mlp_in[t_ids].float()
            gu = F.linear(x_e, w13)  # [M, 2304]
            g = gu[:, 0::2]
            u = gu[:, 1::2]
            act = (self.gate_cap * torch.tanh(g / self.gate_cap)) * torch.sigmoid(g) * (self.up_cap * torch.tanh(u / self.up_cap))
            act = self._bound(act * topk_w[t_ids, k_ids].unsqueeze(-1))
            y_e = F.linear(act, w2)
            moe_out.index_add_(0, t_ids, y_e)

        moe_out = self._bound(moe_out)
        mlp_out = m_sc.step(moe_out, self.weights[lp + "mlp_sconv.weight"])
        mlp_out = self._bound(mlp_out)
        cur_out = self._bound(ffn_inp.float() + mlp_out.float())
        return cur_out

    @torch.no_grad()
    def forward(
        self,
        input_ids: list[int],
        cache: dict[str, Any],
        capture_layers: set[int] | None = None,
        record_trace: bool = False,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor], dict[str, torch.Tensor]]:
        T = len(input_ids)
        ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        positions = torch.arange(cache["seq_len"], cache["seq_len"] + T, dtype=torch.long, device=self.device)
        trace: dict[str, torch.Tensor] = {}
        captures: dict[int, torch.Tensor] = {}

        cur = F.embedding(ids, self.weights["embed.weight"].float())
        cur = self._bound(rms_norm(cur, self.weights["embed_norm.weight"], self.eps))
        if record_trace:
            trace["embed_out"] = cur[-1].cpu().clone()

        blocks: list[torch.Tensor] = []
        il = 0

        # Prelude: ip = 0..5
        for ip in range(0, self.core_start):
            if capture_layers and il in capture_layers:
                captures[il] = cur.clone()
            cur = self._run_layer(il, ip, cur, blocks, cache, positions, None)
            if record_trace and ip in (0, 5):
                trace[f"after_l{il}_p{ip}"] = cur[-1].cpu().clone()
            il += 1

        prelude = cur
        prelude_blocks = list(blocks)
        q_cross = self.weights["middle_cycle_cross_pass_queries"]  # [24, 2304]
        kv_cross = self.weights["middle_cycle_cross_pass_kv_queries"]  # [24, 2304]

        for p in range(self.num_passes):
            bank: list[torch.Tensor] = []
            bank_out: torch.Tensor | None = None
            if p > 0:
                bank = blocks[len(prelude_blocks) + 1 :]
                bank_out = cur
                cur = prelude
                blocks = list(prelude_blocks)
            for ip in range(self.core_start, self.core_end):
                if capture_layers and il in capture_layers:
                    captures[il] = cur.clone()
                messages = None
                if p > 0:
                    idx = ip - self.core_start
                    mq = torch.stack([q_cross[2 * idx], kv_cross[2 * idx], kv_cross[2 * idx + 1], q_cross[2 * idx + 1]], dim=0)
                    messages = block_attn_res([*bank, bank_out], mq, self.eps)  # [T, 4, H]
                cur = self._run_layer(il, ip, cur, blocks, cache, positions, messages)
                if record_trace and ip in (6, 11, 17):
                    trace[f"after_l{il}_p{ip}_pass{p}"] = cur[-1].cpu().clone()
                il += 1

        # Epilogue: ip = 18..23
        for ip in range(self.core_end, self.n_phys):
            if capture_layers and il in capture_layers:
                captures[il] = cur.clone()
            cur = self._run_layer(il, ip, cur, blocks, cache, positions, None)
            if record_trace and ip in (18, 20, 23):
                trace[f"after_l{il}_p{ip}"] = cur[-1].cpu().clone()
            il += 1

        # Output
        h = block_attn_res([*blocks, cur], self.weights["attn_res.output_query"], self.eps)[:, 0, :]
        h = self._bound(rms_norm(h, self.weights["norm.weight"], self.eps))
        if capture_layers and self.n_logical in capture_layers:
            captures[self.n_logical] = h.clone()
        if record_trace:
            trace["final_norm"] = h[-1].cpu().clone()

        logits = F.linear(h[-1:].float(), self.weights["unembed.weight"].float())[0]
        logits = self.softcap * torch.tanh(logits / self.softcap)
        logits = logits[: self.unpadded_vocab]
        cache["seq_len"] += T
        return logits, captures, trace

    # ---- DSpark MTP Reference ----
    def _apply_rope_neox(self, x: torch.Tensor, positions: torch.Tensor, theta: float = 1e6) -> torch.Tensor:
        # x: [T, H, D], positions: [T]
        D = x.shape[-1]
        half = D // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=x.device, dtype=torch.float32) / half))
        freqs = torch.outer(positions.float(), inv_freq)  # [T, half]
        cos = freqs.cos()[:, None, :]
        sin = freqs.sin()[:, None, :]
        x1, x2 = x.float()[..., :half], x.float()[..., half:]
        out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return out

    def _mtp_mix_norm(self, stacked: torch.Tensor, row: int, norm_w: torch.Tensor) -> torch.Tensor:
        # stacked: [T, 4, H]
        alpha = F.softmax(self.mtp["fuse_weight"].float(), dim=-1)[row]  # [4]
        mixed = torch.einsum("tsh,s->th", stacked.float(), alpha)
        return self._bound(rms_norm(mixed, norm_w, self.eps))

    @torch.no_grad()
    def mtp_draft_block(
        self,
        anchor_token: int,
        ctx_captures: torch.Tensor,  # [ctx_len, 4, H]
        block_size: int = 8,
    ) -> list[int]:
        ctx_len = ctx_captures.shape[0]
        ctx_pos = torch.arange(ctx_len, device=self.device)
        blk_pos = torch.arange(ctx_len, ctx_len + block_size, device=self.device)

        # Draft block embeddings: slot 0 = target_embed_norm(anchor), slots 1..7 = mask_embed
        anchor_e = F.embedding(torch.tensor([anchor_token], device=self.device), self.weights["embed.weight"].float())
        anchor_e = self._bound(rms_norm(anchor_e, self.weights["embed_norm.weight"], self.eps))  # [1, H]
        masks = self.mtp["mask_embed"].float().unsqueeze(0).expand(block_size - 1, -1)
        h = torch.cat([anchor_e, masks], dim=0)  # [8, H]

        local_mtp = {0, 2}
        rep = self.n_heads // self.n_kv_heads
        for li in range(3):
            lp = f"layers.{li}."
            # Context KV
            ctx_h = self._mtp_mix_norm(ctx_captures, li, self.mtp[lp + "dflare_norm.weight"])
            k_ctx = self._lin(ctx_h, self.mtp[lp + "self_attn.k_proj_ctx.weight"]).view(ctx_len, self.n_kv_heads, self.head_dim)
            v_ctx = self._lin(ctx_h, self.mtp[lp + "self_attn.v_proj_ctx.weight"]).view(ctx_len, self.n_kv_heads, self.head_dim)
            k_ctx = self._bound(rms_norm(k_ctx, self.mtp[lp + "self_attn.k_norm_ctx.weight"], self.eps))
            k_ctx = self._apply_rope_neox(k_ctx, ctx_pos)

            # Block QKV
            h_ln = self._bound(rms_norm(h, self.mtp[lp + "input_layernorm.weight"], self.eps))
            q = self._lin(h_ln, self.mtp[lp + "self_attn.q_proj.weight"]).view(block_size, self.n_heads, self.head_dim)
            k_b = self._lin(h_ln, self.mtp[lp + "self_attn.k_proj.weight"]).view(block_size, self.n_kv_heads, self.head_dim)
            v_b = self._lin(h_ln, self.mtp[lp + "self_attn.v_proj.weight"]).view(block_size, self.n_kv_heads, self.head_dim)
            q = self._apply_rope_neox(self._bound(rms_norm(q, self.mtp[lp + "self_attn.q_norm.weight"], self.eps)), blk_pos)
            k_b = self._apply_rope_neox(self._bound(rms_norm(k_b, self.mtp[lp + "self_attn.k_norm.weight"], self.eps)), blk_pos)

            if li in local_mtp and ctx_len > self.swa:
                k_ctx = k_ctx[-self.swa :]
                v_ctx = v_ctx[-self.swa :]
            k_all = torch.cat([k_ctx, k_b], dim=0).repeat_interleave(rep, dim=1)
            v_all = torch.cat([v_ctx, v_b], dim=0).repeat_interleave(rep, dim=1)
            # Non-causal across block
            scores = torch.einsum("thd,shd->ths", q, k_all) * (self.head_dim**-0.5)
            probs = F.softmax(scores, dim=-1)
            attn_out = torch.einsum("ths,shd->thd", probs, v_all).reshape(block_size, self.n_heads * self.head_dim)
            h = self._bound(h.float() + self._lin(attn_out, self.mtp[lp + "self_attn.o_proj.weight"]))

            h_mln = self._bound(rms_norm(h, self.mtp[lp + "post_attention_layernorm.weight"], self.eps))
            g = self._lin(h_mln, self.mtp[lp + "mlp.gate_proj.weight"])
            u = self._lin(h_mln, self.mtp[lp + "mlp.up_proj.weight"])
            m_out = self._lin(self._bound(F.silu(g) * u), self.mtp[lp + "mlp.down_proj.weight"])
            h = self._bound(h.float() + m_out)

        h_draft = self._bound(rms_norm(h, self.mtp["norm.weight"], self.eps))
        f_rnn = self._mtp_mix_norm(ctx_captures[-1:], 3, self.mtp["markov_head.dflare_norm.weight"])[0]
        state = torch.zeros(256, dtype=torch.float32, device=self.device)
        prev = anchor_token
        drafted = []
        w_unembed = self.weights["unembed.weight"].float()
        w1 = self.mtp["markov_head.markov_w1.weight"].float()
        rnn_w = self.mtp["markov_head.rnn_proj.weight"].float()
        rnn_b = self.mtp["markov_head.rnn_proj.bias"].float()
        ho_w = self.mtp["markov_head.hidden_out.weight"].float()
        ho_b = self.mtp["markov_head.hidden_out.bias"].float()
        for i in range(1, block_size):
            z = torch.cat([state, w1[prev], h_draft[i], f_rnn], dim=0)
            proj = F.linear(z, rnn_w, rnn_b)
            g_raw, c_raw, o_raw = proj.chunk(3, dim=-1)
            gate = torch.sigmoid(g_raw)
            state = gate * state + (1.0 - gate) * torch.tanh(c_raw)
            latent = torch.tanh(o_raw)
            delta_h = F.linear(latent, ho_w, ho_b)
            logits = F.linear(h_draft[i] + delta_h, w_unembed)[: self.unpadded_vocab]
            tok = int(torch.argmax(logits).item())
            drafted.append(tok)
            prev = tok
        return drafted


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--save", type=Path, default=None)
    ap.add_argument("--test-mtp", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(str(args.model_dir))
    messages = [{"role": "user", "content": "Explain why the sky is blue in one concise sentence."}]
    prompt_text = tok.apply_chat_template(messages, tokenize=False, reasoning_effort=0.0)
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    print(f"Prompt ({len(prompt_ids)} tokens): {prompt_text!r}")
    print(f"Prompt IDs: {prompt_ids}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Inkling2MiniReference(args.model_dir, device=device, bf16_boundaries=True)
    cache = model.init_cache()
    capture_set = {5, 23, 41, 48} if args.test_mtp else None

    t0 = time.time()
    logits, caps, trace = model.forward(prompt_ids, cache, capture_layers=capture_set, record_trace=True)
    topv, topi = torch.topk(logits, 10)
    print(f"Prefill done in {time.time() - t0:.2f}s. Top-10 logits:")
    for v, i in zip(topv.tolist(), topi.tolist()):
        print(f"  id={i:6d} logit={v:8.4f} text={tok.decode([i])!r}")

    all_caps = [torch.stack([caps[l] for l in (5, 23, 41, 48)], dim=1)] if args.test_mtp else []
    generated = [int(topi[0].item())]
    first_logits = logits.cpu().clone()

    for step in range(1, args.max_tokens):
        nxt_logits, step_caps, _ = model.forward([generated[-1]], cache, capture_layers=capture_set, record_trace=False)
        if args.test_mtp:
            all_caps.append(torch.stack([step_caps[l] for l in (5, 23, 41, 48)], dim=1))
        nxt = int(torch.argmax(nxt_logits).item())
        generated.append(nxt)
        if nxt == 200006:
            break

    print(f"\nGenerated {len(generated)} tokens: {generated}")
    print(f"Decoded:\n{tok.decode(generated)}")

    if args.test_mtp and len(generated) >= 10:
        ctx_caps_full = torch.cat(all_caps, dim=0)  # [len(prompt) + len(generated) - 1, 4, H]
        # Test DSpark acceptance at several positions along the generated sequence
        total_drafted = 0
        total_accepted = 0
        for pos_idx in range(0, min(len(generated) - 8, 16), 4):
            ctx_len = len(prompt_ids) + pos_idx
            anchor = generated[pos_idx]
            drafted = model.mtp_draft_block(anchor, ctx_caps_full[:ctx_len], block_size=8)
            truth = generated[pos_idx + 1 : pos_idx + 8]
            acc = 0
            for d, g in zip(drafted, truth):
                if d == g:
                    acc += 1
                else:
                    break
            total_drafted += len(drafted)
            total_accepted += acc
            print(f"DSpark @step {pos_idx} (anchor={anchor}): acc={acc}/7  draft={drafted}  truth={truth}")
        print(f"DSpark mean accepted: {total_accepted / max(1, (total_drafted // 7)):.2f} / 7")

    if args.save is not None:
        torch.save(
            {
                "prompt_text": prompt_text,
                "prompt_ids": prompt_ids,
                "first_logits": first_logits,
                "generated": generated,
                "trace": trace,
            },
            str(args.save),
        )
        print(f"Saved reference oracle to {args.save}")


if __name__ == "__main__":
    main()
