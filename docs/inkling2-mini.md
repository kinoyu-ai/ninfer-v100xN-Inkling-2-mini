# Inkling-2-Mini on Tesla V100 (`sm_70`)

This fork of [`tuxKOH/ninfer-v100xN`](https://github.com/tuxKOH/ninfer-v100xN) was developed and
optimized with **Codex** and **Antigravity (Agy)**. It adds native, lossless support and Volta
(`sm_70`) optimizations for
[`thinkingmachines/Inkling-2-Mini`](https://huggingface.co/thinkingmachines/Inkling-2-Mini)
(`inkling2-mini/nvfp4-mxfp8`), including its **DSpark** multi-token speculative drafter
(`mtp.safetensors`).

## Architecture and precision (zero requantization)

`tools/convert/inkling2_mini/convert.py` preserves every weight tensor bit-for-bit from the
published Hugging Face checkpoint into a single `.ninfer` v2 artifact (`~18.21 GiB`, 748 objects):

| Component | Stored format | Notes |
|---|---|---|
| MoE experts (physical layers 0–19) | **NVFP4** (original) | FP4 E2M1 codes, E4M3 block-16 scales, FP32 `amax`/`denom` |
| MoE experts (physical layers 20–23) | **MXFP8** (original) | FP8 E4M3 codes, UE8M0 block-32 scales |
| Attention, Embed, LM Head, ShortConv, Loop Control | **BF16** (original) | AttnRes queries and router biases stay **FP32** |
| DSpark drafter (`mtp.safetensors`) | **BF16** (original) | 3-layer block drafter + folded Markov RNN |
| KV Cache (SWA-1024 ring + Paged Global) | **BF16** | Full precision, no KV quantization |

The runtime executes all 48 logical layers (prelude 0–5, 3 looped passes over core 6–17 with
cross-pass control gates, epilogue 18–23), Block AttnRes, static/dynamic causal ShortConv (`W=4`),
hybrid SWA-1024 + global attention with learned relative bias, `C=1..8` concurrent batching,
TP2 direct PCIe P2P collectives, CUDA Graphs, and DSpark speculative decoding (`--spec mtp` for
drafts `1..5`, `--spec dflash` for drafts `1..7`).

## Build and artifact conversion

Build with CUDA 12.8 targeting `sm_70`:

```bash
cmake -S . -B build-sm70 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=70
ninja -C build-sm70 ninfer ninfer-serve ninfer_inkling2_ops_test
```

Convert a local copy of `thinkingmachines/Inkling-2-Mini` (including `mtp.safetensors` when present):

```bash
python3 tools/convert/inkling2_mini/convert.py \
  --model-dir /path/to/Inkling-2-Mini \
  --output /path/to/inkling2_mini_nvfp4_mxfp8.ninfer
```

Pass `--no-mtp` if you want an artifact without the DSpark drafter.

## Running CLI and HTTP server (2× V100-16GB, TP2)

```bash
# OpenAI/Anthropic compatible server — high-throughput normal mode (C=1..8)
./build-sm70/apps/ninfer-serve /path/to/inkling2_mini_nvfp4_mxfp8.ninfer \
  --tp 2 --devices 0,1 --max-concurrency 8 --max-context 8192 --port 8080

# Low-latency / code & long-context mode with DSpark speculative decoding (draft=2 or 3)
./build-sm70/apps/ninfer-serve /path/to/inkling2_mini_nvfp4_mxfp8.ninfer \
  --tp 2 --devices 0,1 --spec mtp --draft-tokens 3 \
  --max-concurrency 4 --max-context 8192 --port 8080

# CLI single-prompt run (greedy, no thinking)
./build-sm70/apps/ninfer /path/to/inkling2_mini_nvfp4_mxfp8.ninfer \
  --tp 2 --devices 0,1 --spec mtp --draft-tokens 3 \
  --no-thinking --greedy --max-new 256 --prompt "Explain CUDA warp scheduling."
```

Use `--spec dflash --draft-tokens 7` for draft windows up to 7.

## Measured performance (2× Tesla V100-SXM2-16GB @ 150 W)

**Environment**: 2× Tesla V100-SXM2-16GB (`sm_70`, PCIe P2P enabled via `iommu=pt`, no NVLink,
GPU power capped at **150 W**), CUDA 12.8, CUDA Graphs enabled.

### Apples-to-apples essay benchmark vs `llama.cpp`

Workload: 4 essay prompts (`C=1, 4`) and 8 essay prompts (`C=8`), `max_tokens=256`, `temperature=0.0`, thinking ON, 16-token warmup, 2 repeats, end-to-end wall throughput (`tok/s`).

| Engine / Mode | Weights | C=1 e2e (tok/s) | C=4 agg (tok/s) | C=8 agg (tok/s) |
|---|---|---:|---:|---:|
| **NInfer Normal** | Native NVFP4 + MXFP8 + BF16 (lossless) | **122.10** *(was 104.79, +16.5%)* | **268.98** *(was 238.01, +13.0%)* | **348.83** *(was 313.24, +11.4%)* |
| **NInfer DSpark-2** (`--spec mtp --draft-tokens 2`) | Native NVFP4 + MXFP8 + BF16 (lossless) | **126.04** *(was 106.88, +17.9%)* | **211.87** *(was 175.42, +20.8%)* | **238.72** *(was 224.33, +6.4%)* |
| **NInfer DSpark-3** (`--spec mtp --draft-tokens 3`) | Native NVFP4 + MXFP8 + BF16 (lossless) | **119.46** *(was 102.01, +17.1%)* | **186.80** *(was 166.17, +12.4%)* | — |
| **NInfer DSpark-7** (`--spec dflash --draft-tokens 7`) | Native NVFP4 + MXFP8 + BF16 (lossless) | **88.91** *(was 75.16, +18.3%)* | **120.80** *(was 104.65, +15.4%)* | — |
| `llama.cpp` Normal | `UD-Q4_K_XL` (requantized GGUF) | 30.7 | 78.3 | — |
| `llama.cpp` Normal | `UD-Q8_K_XL` (requantized GGUF) | 29.3 | 55.9 | — |
| `llama.cpp` + DSpark-3 | `UD-Q4_K_XL` / `UD-Q8_K_XL` | 29.3 / 19.8 | 47.7 / 28.5 | — |
| `llama.cpp` + DSpark-7 | `UD-Q4_K_XL` / `UD-Q8_K_XL` | 24.2 / 16.3 | 39.3 / 23.8 | — |

- Compared with `llama.cpp UD-Q4_K_XL` Normal: **3.98×** faster at C=1 Normal (`122.1` vs `30.7`, **4.11×** with DSpark-2 at `126.0`) and **3.44×** faster at C=4 (`269.0` vs `78.3`), while keeping lossless native weights instead of requantized 4-bit GGUF.
- Compared with `llama.cpp UD-Q8_K_XL` Normal: **4.17×** faster at C=1 and **4.81×** faster at C=4.

### Structured / code generation & 4K long-context DSpark showcase (C=1, greedy, `--no-thinking`)

| Workload | Normal | DSpark-2 | DSpark-3 | DSpark-7 |
|---|---:|---:|---:|---:|
| Python dataclass refactor (256 tok, engine decode) | **128.47 tok/s** *(was 106.08, +21.1%)* | **168.90 tok/s** (`2.19` tok/rnd, 59.6%, `12.97 ms/rnd`) | **166.68 tok/s** (`2.41` tok/rnd, 46.9%, `14.46 ms/rnd`) | **145.32 tok/s** (`3.12` tok/rnd, 30.3%, `21.47 ms/rnd`) |
| 4,274-tok needle retrieval (`--max-context 8192`) | **114.24 tok/s** *(was 83.00, +37.6%)* | — | **181.82 tok/s** (`3.10` tok/rnd, *was 139.77, +30.1%*) | — |

- Per-round decode latency dropped to **`7.78 ms/tok`** (Normal C=1), **`12.97 ms/rnd`** (DSpark-2), and **`14.46 ms/rnd`** (DSpark-3); high-acceptance prompts (`~3.0+ tok/rnd`) reach **182–205+ tok/s** decode while preserving lossless weights and exact greedy semantics.
- Cold prefill non-regression spot-check (`512` and `~4,096` tok, C=1): TTFT `0.525 s / 4.679 s` (Normal) and `0.531 s / 4.673 s` (DSpark-3), matching or slightly improving on the full sweep below.

### Cold prefill throughput & TTFT (512–65,536 tokens, C=1)

Measured via `ninfer-serve --tp 2 --devices 0,1 --max-concurrency 1 --max-context 70000 --no-thinking --greedy --request-log-jsonl`
after one 1,024-token warmup request (`max_tokens=1`, verified `prefix_cache_hit_tokens = 0` on every run).
Values report mean engine `prefill tok/s` (`computed_prefill_tokens / timings_seconds.prefill`) and mean
server `TTFT` across 3 repeats (`512–16,385` tokens) or 2 repeats (`32,768–65,536` tokens):

| Prompt tokens | Normal (`chunk=1024`) | Normal (`chunk=2048`) | DSpark-3 (`chunk=1024`) |
|---:|---:|---:|---:|
| 512 | **1,033.73 tok/s** (0.495 s / TTFT 0.498 s) | — | 1,001.50 tok/s (0.511 s / TTFT 0.516 s) |
| 1,023 | **1,014.45 tok/s** (1.008 s / TTFT 1.012 s) | — | 982.23 tok/s (1.042 s / TTFT 1.049 s) |
| 2,048 | **940.83 tok/s** (2.177 s / TTFT 2.184 s) | 928.26 tok/s (2.206 s / TTFT 2.216 s) | 907.30 tok/s (2.257 s / TTFT 2.268 s) |
| 4,096 | **882.80 tok/s** (4.640 s / TTFT 4.653 s) | 876.14 tok/s (4.675 s / TTFT 4.701 s) | 850.95 tok/s (4.814 s / TTFT 4.834 s) |
| 8,192 | **810.48 tok/s** (10.108 s / TTFT 10.134 s) | 809.25 tok/s (10.123 s / TTFT 10.179 s) | 784.61 tok/s (10.441 s / TTFT 10.471 s) |
| 16,385 | 703.48 tok/s (23.292 s / TTFT 23.350 s) | **709.14 tok/s** (23.106 s / TTFT 23.178 s) | 684.23 tok/s (23.947 s / TTFT 24.005 s) |
| 32,768 | 566.44 tok/s (57.850 s / TTFT 57.994 s) | **576.55 tok/s** (56.835 s / TTFT 56.955 s) | 554.22 tok/s (59.124 s / TTFT 59.217 s) |
| 65,536 | 406.50 tok/s (161.221 s / TTFT 161.473 s) | **417.78 tok/s** (156.868 s / TTFT 157.096 s) | 401.16 tok/s (163.365 s / TTFT 163.620 s) |

- Default `--prefill-chunk 1024` is fastest at 512–8,192 tokens (**810–1,034 tok/s**); `--prefill-chunk 2048`
  pulls ahead at ≥16K (**709.1 / 576.6 / 417.8 tok/s** at 16K / 32K / 64K, up to +2.8% faster at 64K).
- Enabling DSpark-3 (`--spec mtp --draft-tokens 3`) adds only **1.3%–3.6%** prefill overhead for hidden-state
  capture and 3-layer drafter KV population across the full 512–65,536 token sweep.
- Run-to-run variance across repetitions was under ±0.5% at every length; the smooth decline from 1,034 to
  407–418 tok/s reflects quadratic BF16 global attention over the 12 global logical layers (the other 36
  logical layers stay bounded to SWA-1024).

## Key Volta `sm_70` optimizations

- **Fused BF16 projections (`src/ops/inkling2/bf16_linear.cu`)**: coalesced 2-deep prefetched SIMT
  kernels (`1rpw_pf2`, `16tpr_pf2`, `t9_16_16tpr_pf2`), 3-way (`Q/gate/R/sink + K + V`) and
  global-layer 5-way (`+ k_sconv_proj + v_sconv_proj`) projection fusion, fused monotonic softcap
  unembed argmax (`unembed_small_t_argmax_pf2`, reaching `883 GB/s` at `T=1`), and 4-row/warp
  strided Markov logits argmax.
- **Zero-spill coalesced sparse MoE (`src/ops/inkling2/sparse_moe.cu`)**: single-wave 64-CTA
  Last-CTA fused router (`moe_router_fused` with register-only `warp_select_top8_128`, fused FP16
  input staging, and `accum_f32` zeroing), 2-deep prefetched `T=1` NVFP4/MXFP8 kernels, SMEM-staged
  `TileM=4` grouped `w13` (`18 KB` SMEM, specialized `mc=1/2/3..4`, 2-deep prefetch), and
  128-row/CTA grouped `w2` (`0 B` stack/spills verified via `ptxas -v` on every hot kernel).
- **Fused TP2 packet-push AllReduce + ShortConv (`src/ops/inkling2/sconv.cu`)**: fuses TP2 direct
  P2P two-epoch packet reduction, optional FP32→BF16 cast from MoE `accum_f32`, `attn_sconv` /
  `mlp_sconv`, and residual/block update into a single kernel for **both** Decode (`T=1..8`) and
  Verify (`V=2..8`), plus fused TP2 packet AllReduce + residual add for DSpark MTP layers.
- **Attention, loop-control, and DSpark fusion (`attention.cu`, `attn_res_control.cu`, `dspark.cu`)**:
  single-launch adaptive Last-CTA Q-head Split-KV for both main attention and DSpark block attention
  (zero reduce launches, zero scratch at short context, cached global page lookups), K/V SConv fused
  directly into `attn_prologue`, fused AttnRes + loop-control prep (`attn_res_mix_and_prep`,
  `attn_res_mix_pair_and_prep_qkv`, `swap_pass_banks`), batched 3-layer MTP context mix/KV-write,
  fused MTP final-norm/extract/zero, and zero-HBM-intermediate ping-pong `mtp_markov_rnn_step`.

## Verification and limitations

- **Operator tests**: `./build-sm70/tests/ninfer_inkling2_ops_test` — **164/164 PASS** against
  independent FP64 oracles, TP1/TP2 shard-equivalence, two-epoch packet loopback, long-context
  adaptive Split-KV (`1,100–1,200` tokens across SWA ring wrap & permuted global pages), and fused
  vs unfused parity checks, with **0 B stack / 0 spills** across all hot Volta kernels.
- **Exact token parity vs PyTorch (`tools/reference/inkling2_mini/reference.py`)**: **32/32 exact
  token match** across `TP2 Graph Normal`, `TP2 Eager Normal`, `TP2 Graph DSpark-2`,
  `TP2 Graph DSpark-3`, `TP2 Graph DSpark-5`, and `TP2 Graph DSpark-7`.
- **Tokenizer parity vs HF `AutoTokenizer`**: exact prompt token count match (0 delta) across
  English contractions, Japanese, Python code, numbers/URLs, and special-character inputs.
- **Long-context & concurrency**: verified exact needle retrieval at 4,184 prompt tokens
  (exercising >4 SWA-1024 ring wraps and >64 paged global KV pages), plus `C=8` concurrent
  greedy and stochastic (`temperature=0.7, top_p=0.95`) stability.
- **V100-32GB×2 and V100-32GB×1 compatibility**:
  - **Dynamic VRAM & KV planning**: No 16 GB VRAM cap is hardcoded. On 2× V100-32GB (~22 GiB/GPU
    free after weights), full native **262,144-token** context (`~4.0 GiB/GPU` Normal or
    `~4.5 GiB/GPU` DSpark paged BF16 global KV) and concurrent `C=8` long-context serving fit with
    ample headroom.
  - **Interconnect portability**: Startup (`ops::enable_peer_access`) automatically qualifies
    direct **NVLink P2P**, direct **PCIe P2P**, or **non-P2P** fallback (verified CUDA UVA D2D
    staging / WSL2 mapped-host packets). Fused TP2 packet-push AllReduce + ShortConv activates
    whenever direct P2P packet mailboxes are allocated and cleanly falls back to standard
    `allreduce_sum` + `sconv_static_decode` otherwise.
  - **TP1 (`--tp 1`)**: Resident weights require `~17.66 GiB`, which exceeds a 16 GB V100 (TP2
    required on 16 GB) but fits on a single 32 GB V100. TP1 sharding, bindings, and operator paths
    are implemented and verified by unit tests (`ninfer_inkling2_ops_test`), though whole-model
    benchmarks so far were measured on 2× V100-SXM2-16GB (`--tp 2`).
- **Limitations**:
  - **Batched FP associativity (`C >= 2`)**: grouped MoE `w2` accumulates via `atomicAdd`, so
    multi-request greedy runs can diverge late on near-tied logits relative to `C=1`.
  - **Downstream benchmarks**: exact PyTorch parity and FP64 operator correctness are verified;
    no separate GPQA/MMLU evaluation was run.
