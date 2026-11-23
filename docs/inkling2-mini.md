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

Workload: 4 essay prompts (`C=1, 4`, `bench_ninfer.py`), `max_tokens=256`, `temperature=0.0`, thinking ON, 16-token warmup, 2 repeats (mean), end-to-end wall throughput (`tok/s`). `C=8` uses 8 distinct essay prompts with the same settings. "was" values are the original pre-optimization baseline.

| Engine / Mode | Weights | C=1 e2e (tok/s) | C=4 agg (tok/s) | C=8 agg (tok/s) |
|---|---|---:|---:|---:|
| **NInfer Normal** | Native NVFP4 + MXFP8 + BF16 (lossless) | **122.0** *(was 104.8)* | **268.2** *(was 238.0)* | **356.9** |
| **NInfer Normal (`temp=0.7, top_p=0.95`)** | Native NVFP4 + MXFP8 + BF16 (lossless) | **121.1** | **267.3** | — |
| **NInfer DSpark-2** (`--spec mtp --draft-tokens 2`) | Native NVFP4 + MXFP8 + BF16 (lossless) | **127.3** *(was 106.9)* | **208.9** *(was 175.4)* | **241.1** |
| **NInfer DSpark-3** (`--spec mtp --draft-tokens 3`) | Native NVFP4 + MXFP8 + BF16 (lossless) | **122.0** *(117.9 / 126.2; was 102.0)* | **189.9** *(was 166.2)* | — |
| **NInfer DSpark-7** (`--spec dflash --draft-tokens 7`) | Native NVFP4 + MXFP8 + BF16 (lossless) | **94.5** *(was 75.2)* | **119.4** *(was 104.7)* | — |
| `llama.cpp` Normal | `UD-Q4_K_XL` (requantized GGUF) | 30.7 | 78.3 | — |
| `llama.cpp` Normal | `UD-Q8_K_XL` (requantized GGUF) | 29.3 | 55.9 | — |
| `llama.cpp` + DSpark-3 | `UD-Q4_K_XL` / `UD-Q8_K_XL` | 29.3 / 19.8 | 47.7 / 28.5 | — |
| `llama.cpp` + DSpark-7 | `UD-Q4_K_XL` / `UD-Q8_K_XL` | 24.2 / 16.3 | 39.3 / 23.8 | — |

- Compared with `llama.cpp UD-Q4_K_XL` Normal: **3.97×** faster at C=1 Normal (`122.0` vs `30.7`, **4.15×** with DSpark-2 at `127.3`) and **3.43×** faster at C=4 (`268.2` vs `78.3`), while keeping lossless native weights instead of requantized 4-bit GGUF.
- Compared with `llama.cpp UD-Q8_K_XL` Normal: **4.16×** faster at C=1 and **4.80×** faster at C=4.
- Wave 5 versus the Wave 4.5 build (`b6dc2214`) is within the ±3% run-to-run spread of these
  150 W measurements on short-context essays (for example Normal C=1 `122.1 → 122.0`, DSpark-2
  C=4 `211.9 → 208.9`). Non-greedy sampling now runs inside CUDA Graphs with a direct BF16
  unembed, but that measured only `120.3 → 121.1 tok/s` (C=1) and `267.9 → 267.3 tok/s` (C=4):
  this TP2 decode is GPU-bound, so removing eager launches and the FP32 logit round-trip is not
  visible end to end.

### Structured / code generation & 4K–16K long-context DSpark showcase (C=1, greedy, `--no-thinking`)

| Workload | Normal | DSpark-2 | DSpark-3 | DSpark-7 |
|---|---:|---:|---:|---:|
| Python dataclass refactor (256 tok, engine decode; Wave 4.5 build) | **128.47 tok/s** *(was 106.08, +21.1%)* | **168.90 tok/s** (`2.19` tok/rnd, 59.6%, `12.97 ms/rnd`) | **166.68 tok/s** (`2.41` tok/rnd, 46.9%, `14.46 ms/rnd`) | **145.32 tok/s** (`3.12` tok/rnd, 30.3%, `21.47 ms/rnd`) |
| 4,274-tok needle retrieval (`--max-context 8192`; Wave 4.5 build) | **114.24 tok/s** *(was 83.00, +37.6%)* | — | **181.82 tok/s** (`3.10` tok/rnd, *was 139.77, +30.1%*) | — |
| 16,304-tok long-context decode (`--max-context 20480`, 128 new; Wave 4.5 → Wave 5) | **95.92 → 99.32 tok/s** (single run) | — | **119.10 → 131.78 tok/s**; `20.74 → 20.11 ms/rnd` (−3.0%), acceptance `2.47 → 2.65` tok/rnd | — |

- The 16K DSpark-3 gain is mostly a change in draft acceptance (`2.47 → 2.65` tok/rnd), because
  the single-exp online softmax changes floating-point rounding in global attention; the
  per-round kernel time improves by about 3%. Both outputs are valid greedy decodes of their own
  numerics, not bit-identical to each other.

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

- **Fused BF16 projections & direct BF16 unembed (`src/ops/inkling2/bf16_linear.cu`, `program.cpp`)**:
  coalesced 2-deep prefetched SIMT kernels (`1rpw_pf2`, `16tpr_pf2`, `t9_16_16tpr_pf2`), 3-way
  (`Q/gate/R/sink + K + V`) and global-layer 5-way (`+ k_sconv_proj + v_sconv_proj`) projection
  fusion, fused monotonic softcap unembed argmax (`unembed_small_t_argmax_pf2`, reaching `883 GB/s`
  at `T=1`), direct BF16 fused softcap+mask unembed epilogue (`unembed_bf16_{1rpw,small_t_16tpr}_pf2`,
  eliminating FP32 logit HBM round-trips, 1-CTA softcap scan, and `cast_fp32_to_bf16`) captured inside
  cross-device `decode_logits_graphs_` for non-greedy sampling, multi-CTA `softcap_mask_f32_kernel`,
  and 4-row/warp strided Markov logits argmax.
- **Zero-spill coalesced sparse MoE (`src/ops/inkling2/sparse_moe.cu`)**: single-wave 64-CTA
  Last-CTA fused router (`moe_router_fused` with register-only `warp_select_top8_128`, free
  Warps 1–3 `T=1` `x_staged` conversion overlapped with Warp 0 top-8 selection, and `T=2..8`
  input staging + `accum_f32` zeroing), 2-deep prefetched `PreStaged` `T=1` NVFP4/MXFP8 kernels,
  SMEM-staged `TileM=4` grouped `w13` (`18 KB` SMEM, specialized `mc=1/2/3..4`, 2-deep prefetch), and
  128-row/CTA grouped `w2` (`0 B` stack/spills verified via `ptxas -v` on every hot kernel).
- **Fused TP2 packet-push AllReduce + ShortConv (`src/ops/inkling2/sconv.cu`)**: fuses TP2 direct
  P2P two-epoch packet reduction, optional FP32→BF16 cast from MoE `accum_f32`, `attn_sconv` /
  `mlp_sconv`, and residual/block update into a single kernel for **both** Decode (`T=1..8`) and
  Verify (`V=2..8`, row-major coalesced Phase 1, bank-conflict-free SMEM Phase 2, compile-time
  `FixV` for `V=3,4,6,8`), plus fused TP2 packet AllReduce + residual add for DSpark MTP layers.
- **Attention, loop-control, and DSpark fusion (`attention.cu`, `attn_res_control.cu`, `dspark.cu`)**:
  single-launch adaptive Last-CTA Q-head Split-KV for both main attention and DSpark block attention
  (zero reduce launches, zero scratch at short context, cached global page lookups, and a
  warp-uniform single-exp online softmax for `T>=2` global attention once a split holds at least
  160 keys), K/V SConv
  fused directly into `attn_prologue`, fused AttnRes + loop-control prep (`attn_res_mix_and_prep`,
  `attn_res_mix_pair_and_prep_qkv`, `swap_pass_banks`), batched 3-layer MTP context mix/KV-write,
  fused MTP final-norm/extract/zero, and zero-HBM-intermediate ping-pong `mtp_markov_rnn_step`.

## Verification and limitations

- **Operator tests**: `./build-sm70/tests/ninfer_inkling2_ops_test` — **172/172 PASS** against
  independent FP64 oracles, bit-exact direct-BF16 unembed & `T=1` PreStaged MoE checks, TP1/TP2
  shard-equivalence, two-epoch packet loopback, long-context adaptive Split-KV (`1,100–1,200`
  tokens across SWA ring wrap & permuted global pages, plus `4,096` / `4,100`-token global cases
  that exercise the single-exp online softmax route), and fused vs unfused parity checks, with
  **0 B stack / 0 spills** across all hot Volta kernels.
- **Exact token parity vs PyTorch (`tools/reference/inkling2_mini/reference.py`)**: **32/32 exact
  token match** across `TP2 Graph Normal`, `TP2 Eager Normal`, `TP2 Graph DSpark-2`,
  `TP2 Graph DSpark-3`, `TP2 Graph DSpark-5`, `TP2 Graph DSpark-7`, and the Graph-captured Direct
  BF16 sampling path (`--temperature 0.01`). A 48-token stochastic run
  (`--temperature 0.7 --top-p 0.95 --seed 42`) matched between Graph and eager; longer runs are
  subject to the run-to-run nondeterminism described below.
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
  - **Run-to-run floating-point nondeterminism**: grouped MoE `w2` accumulates via `atomicAdd`
    for every multi-token pass (prefill chunks, DSpark verification, and `C >= 2` decode). Even a
    single C=1 request can therefore diverge late on near-tied logits between two identical runs
    (observed at generated token 45–59 of a 64–128-token greedy run). The first 32 tokens of the
    PyTorch oracle prompt are stable.
  - **Downstream benchmarks**: exact PyTorch parity and FP64 operator correctness are verified;
    no separate GPQA/MMLU evaluation was run.
