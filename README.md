# NInfer V100X2

[中文文档](README.zh-CN.md) · [Performance methodology](docs/performance.md)

Experimental native RAM-KV is opt-in with `--ram-kv-window N` (27B SM70, TP1/TP2, one Text/MTP
request). It stores packed history in RAM and uses training-free **lexical retrieval**, not KVMem's
Q/K-vector retriever. It changes long-history attention, including prefill; default full history
is unchanged. See [RAM-KV contracts](docs/maintainer/paged-kv-cache.md#15-optional-native-ram-kv-experiment)
and [launch options](docs/cli.md). The external KVMem evaluation is not a native NInfer speed claim.

Performance results are separated into [P2P enabled](#p2p-enabled) and
[P2P disabled](#p2p-disabled); build and launch instructions are shared below.

This fork is tuned for one-request Qwen3.8-27B inference on **2 × Tesla V100-SXM2 16 GB**
(`sm_70`, CUDA 12.8). **QUASAR NVFP4 v3 is the recommended model for text inference**, with
TP2, complete INT8 group-64 KV and CUDA Graphs. Use MTP3 at 180,000-token capacity for general
tasks; DFlash7 at 98,304-token capacity is an alternative for high-acceptance structured output.
Context capacity is an allocation limit, not the number of prompt tokens in a benchmark.
The convenience launcher's unmodified default is still the LM Studio Q4_K_M-derived artifact;
the [launch commands](#build-and-run) explicitly select QUASAR.

The starting point combines the RTX 3060 TP2 work and the Volta implementation from
[`geoffwatts/ninfer-v100`](https://github.com/geoffwatts/ninfer-v100), based on
[Neroued/ninfer](https://github.com/Neroued/ninfer). This README focuses on V100X2-specific changes
and measurements, with external source-checkpoint quality evaluations labeled separately.
Inherited RTX 5090/Ampere/Ada results and general upstream capabilities are intentionally omitted.

## Recommended model: QUASAR NVFP4

Use the [QUASAR NInfer artifact, release v3](https://huggingface.co/MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer/tree/v3).
On this host, [measured QUASAR throughput](#quasar-nvfp4) reaches **139.80 tok/s** on a
3,072-token coding input and **100.34 tok/s** after warming an 85,000-token coding input
(MTP3, committed wall decode). DFlash7 reaches **249.00 tok/s** on the native 32-record JSONL
task, or **153.12 tok/s** with 85,000 input tokens. These are workload-specific results;
DFlash is not faster on every task.

Maintainer's personal experience: "It feels in the same tier as FP8."

Public **source-checkpoint quality** results support near-BF16 task accuracy, not mathematical
losslessness. The [QUASAR authors' model card](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4#quality-and-size-comparison)
reports GPQA-D over two runs (396 answers) and AIME'26 over three repeats (90 answers):

| Source checkpoint | GPQA-D (%) | AIME'26 (%) |
|---|---:|---:|
| BF16 original | 91.41 | 100.0 |
| QUASAR NVFP4 | 90.91 | 100.0 |

[Rieker's independent comparison](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/192)
uses a DGX Spark GB10 with vLLM, FP8 KV and MTP5, not this V100 runtime. MBPP/HumanEval/GSM8K
use thinking off, with 257/164/500 cases; PPL uses 323 windows and text KLD uses 49 prompts:

| Source checkpoint | MBPP (%) | HumanEval (%) | GSM8K (%) | PPL ↓ | Text KLD ↓ |
|---|---:|---:|---:|---:|---:|
| BF16 original | 70.8 | 93.3 | 97.0 | 7.993 | reference |
| Official FP8 | 69.3 | 95.1 | 97.4 | 8.029 | 0.0117 |
| QUASAR NVFP4 | 68.9 | 93.9 | 96.8 | 8.247 | 0.0682 |

Task scores are close to BF16, but distribution fidelity is not equal to FP8. Neither public
evaluation measures this converted `.ninfer` artifact on V100; broad quality equivalence is
not established by our inference checks. V100 dequantizes these weights without A4 activation
quantization. V3 is a container upgrade, not a new QAT checkpoint.

**Limitation: QUASAR is text-only in this fork.** The source checkpoint and published container
include Vision, but this runtime rejects QUASAR image/video requests and uploads zero Vision
weights. MTP3 uses 8.66 GiB of weights per card; the full artifact occupies 18.42 GiB on disk.
Official NVFP4 and GGUF-derived Q4_K_M remain supported alternatives.

TP2 Vision is an **experimental, opt-in feature** from [PR #1](https://github.com/tuxKOH/ninfer-V100X2/pull/1).
It is disabled by default and does not enable QUASAR Vision. Its visual-prefix bridge test
fails on this host because cached and cold prefill produce different greedy continuations;
the cause remains unresolved. Use the [cold-request Vision server profile](#experimental-tp2-vision)
with `--no-prefix-reuse` to avoid this unqualified cache path. Enabling Vision requires extra
GPU memory; TP4 Vision and DFlash plus Vision remain unsupported.

## V100X2 changes

- **Q4_K_M prefill:** cooperative SM70 GGML-K block decoding materializes rows into caller-owned
  FP16 workspace and feeds the Volta CUTLASS Tensor-Core GEMM. GDN control projection has an FP32
  output path. The original GGUF Q4_K/Q6_K codes and scales are kept unchanged.
- **Volta FP8/NVFP4 execution:** wide prefill projections decode weights once per call and use
  SM70 CUTLASS, including the prepacked TP2 matrices. NVFP4 decoding reads complete K16 tuples
  cooperatively. Wide TP2 NVFP4 MLP calls at T≥2048 use a 256-column GEMM tile.
  Decode uses QPN Tensor-Core kernels. Both wide prefill and decode keep gate/up
  projections in FP32 through SwiGLU, rounding only the final activation to BF16.
  The prepacked FP8 output head now uses its matching kernel for single-token calls and chunk
  tails too; reading that layout through row-major GEMV was a correctness bug. NVFP4 A4 is not
  supported on V100.
- **Long-context decode:** the split-KV attention reducer shares softmax weights and reads eight
  split vectors cooperatively, then accumulates in the original FP32 order. It retains every
  history token, the same KV format and the same partial-output precision. The INT8 attention
  kernel also computes each QK score tile once and shares it across the four output-dimension
  warps, preserving their softmax and PV accumulation order.
- **DFlash sliding-window attention:** the Volta path uses FP32 split numerators and a warp-per-row
  route once enough rows are in flight; direct tiny windows retain the CTA route. The supported
  local window is 2,048 or 4,096 positions, and the complete-history target KV is unchanged.
- **GDN prefill:** long normalized inputs prepare each Q/K row once in FP32, avoiding repeated
  normalization across state tiles. The sequential FP32 state transition is unchanged. The
  3,072-token TP2 GDN scratch allocation is 24 MiB.
- **TP2 transfers:** collectives use graph-capturable UVA device-to-device copies and two event
  pairs. Startup checks exact transfer bytes in both directions. This host now uses direct
  PCIe P2P after rebooting with `iommu=pt` and identity IOMMU domains. On translated `DMA`/`DMA-FQ`
  domains, startup instead selects verified CUDA-managed staging for PCIe peers. An active
  NVLink mesh is identified from NVML's actual remote PCI endpoints and qualified separately;
  it bypasses the host-IOMMU restriction without changing system settings. Neither path maintains a
  second explicit pinned-host copy route; PCIe P2P is not NVLink. Qualified SM70 all-reduces
  of at most 80 KiB now sum direct peer reads into rank-local staging; both reads retire before
  either input is overwritten. Wide prefill and peer-off collectives keep the DMA route.
- **Volta TP2 residuals:** Q5 row shards now select executable SM70 SIMT, fused MMA, or CUTLASS
  routes with the actual peak workspace reserved. BF16 row shards use the SM70 CUTLASS route.
  Both paths pass their independent FP64 operator checks and the two-card composition tests.
- **NVFP4 v3 container:** the reader accepts the upstream `NINFER\x00\x03` single-file
  `qwen3.8-27b` container and projects its physical objects and logical bindings into the
  registered `qwen3.8-27b/nvfp4` identity without repacking weight bytes. The v3
  embedded chat template is preserved verbatim. Text and MTP load through the normal Engine path;
  v3 Vision objects are projected for the existing Vision route. Its optional
  five-layer DFlash2 package is also bound for text-only TP2 experiments.
- **MTP and DFlash:** the Q4_K_M target uses the native MTP route. An optional five-layer BF16
  auxiliary / W8-projection DFlash2 route is integrated with TP2. Greedy MTP and DFlash target
  verification exchange shard argmax values/IDs instead of full logits; DFlash proposal selection
  merges each rank's exact top-16 keys. Sampled or mixed-batch verification retains the full-logit
  route. Removing DFlash's unused
  full-history draft KV pool leaves the all-local v3 route at 98,304-token capacity; 180,000-capacity
  DFlash is not claimed on two 16-GB cards. Narrow BF16 draft projections use direct BF16
  operands with FP32 SIMT accumulation on V100. MTP remains the default.

## Measurements

Local results below are from this V100X2 host; the separately labeled NVLink deployment uses
a different four-GPU machine. Decode tok/s counts committed output tokens,
not drafted tokens; occupied prompt length is reported separately from maximum capacity.
The two areas distinguish the current direct-P2P configuration from the peer-off control and
earlier staging measurements. NVFP4 and Q4_K_M are different weight artifacts, not a matched
quantization-quality comparison.

### NVLink deployment: TP2 stage

Measured on 2026-10-02 on a second machine with four Tesla V100-SXM2 16 GB cards, all pairwise
NV2, 300 W per card, driver 580.178.04 and CUDA 12.8 / SM70. **NInfer uses only devices 0,1
at TP2 in this table.** The other two GPUs remain idle. Direct NVLink P2P is qualified despite
DMA-FQ host IOMMU domains; no GRUB, driver or reboot changes were needed.

Official Qwen3.8-27B NVFP4 v3, INT8 group-64 KV, 180000 context capacity (180032 allocated),
chunk 2560, greedy optimized MTP3 and CUDA Graphs. Two cold requests per row, no prefix reuse
or extra request warmup; loading and graph priming are excluded. Each request produces one
prefill token plus 512 committed timed decode tokens. Saved prompts retain the final coding
task and assistant suffix. Mean ± sample standard deviation:

| Actual input tokens | Prefill tok/s | Committed decode tok/s | Wall decode tok/s | MTP acceptance |
|---:|---:|---:|---:|---:|
| 3072 | 2101.49 ± 4.03 | 111.19 ± 0.01 | 111.10 ± 0.01 | 72.16% |
| 8192 | 2114.14 ± 6.82 | 109.04 ± 0.03 | 108.91 ± 0.03 | 70.67% |
| 16384 | 2050.33 ± 1.22 | 104.46 ± 0.02 | 104.34 ± 0.02 | 69.00% |
| 32768 | 1887.71 ± 0.55 | 101.99 ± 0.07 | 101.89 ± 0.08 | 74.47% |
| 65536 | 1604.61 ± 3.27 | 90.28 ± 0.03 | 90.23 ± 0.03 | 76.39% |
| 85000 | 1463.73 ± 4.35 | 85.53 ± 0.04 | 85.50 ± 0.05 | 77.11% |

All repetitions reproduce all 513 output IDs and speculative counters, with no EOS/EOG in the measured window.
These fixed coding windows are performance measurements, not scored complete solutions or a
matched motherboard/NVLink A/B. Reports reside on that machine under
`results/tp2-nvlink/`; [bench_tp.py](tools/v100/bench_tp.py) runs the occupancy, capacity and
plain-decode matrices sequentially through `ninfer_bench`.

Capacity-only sweep: the same **512-token input**, chunk 1024, MTP3, two repetitions and
**256 timed decode tokens** at every capacity (512 timed outputs would overflow the 1024 row).
Every row allocates exactly its requested capacity and reproduces all 257 output IDs and
speculative counters. Acceptance is 70.33% throughout; these rows are not long-context inputs.

| Context capacity | Prefill tok/s | Committed decode tok/s | Wall decode tok/s |
|---:|---:|---:|---:|
| 1024 | 1669.31 ± 29.44 | 112.60 ± 0.04 | 112.56 ± 0.03 |
| 2048 | 1676.79 ± 30.19 | 112.86 ± 0.01 | 112.82 ± 0.01 |
| 4096 | 1666.35 ± 33.44 | 112.70 ± 0.02 | 112.66 ± 0.02 |
| 8192 | 1670.13 ± 31.18 | 112.71 ± 0.02 | 112.67 ± 0.02 |
| 16384 | 1669.02 ± 34.89 | 112.91 ± 0.03 | 112.86 ± 0.03 |
| 32768 | 1668.32 ± 37.79 | 112.81 ± 0.04 | 112.75 ± 0.03 |
| 65536 | 1670.67 ± 27.88 | 112.97 ± 0.01 | 112.90 ± 0.02 |

The ordinary-decode control uses the same saved inputs, capacity 180000, chunk 2560 and
512 timed outputs, but `--spec none --draft-tokens 0` without the draft head:

| Actual input | Plain prefill tok/s | Plain decode tok/s | MTP3 decode tok/s |
|---:|---:|---:|---:|
| 3072 | 2132.37 ± 8.53 | 39.584 ± 0.004 | 111.19 ± 0.01 |
| 85000 | 1480.16 ± 4.93 | 28.349 ± 0.003 | 85.53 ± 0.04 |

Within each control the repeated output IDs agree. MTP/plain outputs match at 85K, but not
at 3K; these speed ratios do not prove universal token parity or a no-quality-loss result.
The [phase accounting](docs/performance.md#separate-nvlink-deployment-tp2-stage) separates
model loading, prompt preparation, prefill and committed decode.

### NVLink deployment: NInfer TP4

Measured on 2026-10-03 on the same four-card NV2-linked V100 host above, with all four
devices active. Official Qwen3.8-27B NVFP4 v3, capacity 180000 (180032 allocated), INT8
group-64 KV, chunk 2560, greedy optimized MTP3 and CUDA Graphs. Two cold requests per
point, one prefill output plus 512 committed timed decode outputs, no prefix reuse or
extra warmup. Load/graph priming are outside request timing. Mean ± sample standard deviation:

| Actual input tokens | Prefill tok/s | Committed decode tok/s | Wall decode tok/s | MTP acceptance |
|---:|---:|---:|---:|---:|
| 3072 | 2381.83 ± 37.81 | 125.68 ± 0.08 | 125.42 ± 0.08 | 73.13% |
| 8192 | 2415.01 ± 15.32 | 136.81 ± 0.07 | 136.61 ± 0.07 | 80.62% |
| 16384 | 2381.76 ± 9.29 | 124.09 ± 0.08 | 123.82 ± 0.09 | 72.31% |
| 32768 | 2279.47 ± 2.72 | 117.96 ± 0.02 | 117.73 ± 0.02 | 74.47% |
| 65536 | 2072.25 ± 0.66 | 109.12 ± 0.13 | 108.99 ± 0.13 | 79.30% |
| 85000 | 1957.93 ± 2.16 | 102.60 ± 0.08 | 102.51 ± 0.08 | 78.56% |

Every pair reproduces all 513 output IDs and speculative counters, without EOS/EOG. At
85K, this is 33.76% more prefill and 19.95% more decode throughput than the earlier TP2
snapshot. Acceptance and arithmetic grouping differ, so this is not isolated TP-width scaling
or proof of identical continuations across widths. Reports are in `results/tp4-nvfp4/` on
the remote host; these fixed windows are not scored complete code solutions.

Four-card NVFP4 graph/eager, sampled routing, B=2 MTP and 64 strict teacher-forced positions
pass, with zero argmax disagreements. Prefix checkpoint replay, repeated append, exact
frontiers and partial-round truncation pass exact state controls. A cold-vs-cached choice at
position 28 was a BF16 tie (zero logit deficit); stop-truncation and output-budget truncation
produce identical append outputs, logits and acceptance. Cold re-prefill is not claimed
bit-identical to retained-state execution.

### Four-card 1Cat/vLLM deployment control

For the requested friend-host setup, the isolated launchers under `/home/z/1cat-vllm-v100`
used all four NV2-linked V100-SXM2 cards, the existing OneCat Studio model directories and
loopback ports 6100/6101. The original OneCat UI on port 8888 was not modified. With
180000 maximum context, MTP3 and a 1,570-token benchmark prompt plus 256-token stream:

| 1Cat model | TTFT | streamed decode |
|---|---:|---:|
| Qwen3.8-27B NVFP4 | 0.955 s | 83.03 tok/s |
| Qwen3.8-27B FP8 | 1.590 s | 83.23 tok/s |

These are single short requests through the HTTP streaming client, not 85K occupied-context
measurements and not NInfer TP4 results. The launchers use the model's own FP8/NVFP4 kernels;
NInfer's native block-scaled FP8 identity is measured separately. After the control runs, the
1Cat processes were stopped and the NInfer TP4 API was started on loopback 6200.

The deployed TP4 API launch is `bash tools/v100/serve-tp4.sh /home/z/Models/ninfer/qwen3_8_27b_nvfp4.ninfer`:
loopback port 6200, all four NVLink cards, 180K capacity, MTP3, one active request, 65536
default output budget and prefix reuse. Text-only OpenAI/Responses/Anthropic requests,
unsupported-image rejection and turn-checkpoint replay pass.
The friend can connect through an SSH tunnel; the API is not exposed on a public interface:

```bash
ssh -N -L 6200:127.0.0.1:6200 -p 12031 z@YOUR_HOST
curl http://127.0.0.1:6200/health
```

For a graphical launcher, run `bash tools/v100/ninfer-gui.sh`. The PyQt console exposes the
artifact, TP/device mapping, context/KV capacity, prefill chunk, MTP/DFlash draft window,
CUDA Graph/prefix reuse, concurrency, output limit and sampling defaults, while validating
unsupported combinations before starting `ninfer-serve`. See [`docs/gui.md`](docs/gui.md).

**NVFP4 and native block-128 FP8 TP4 model/state gates, prefix replay, occupancy/capacity
controls and the TP4 HTTP deployment are complete.**

Native FP8 TP4 occupancy (same 180K capacity, INT8 KV, MTP3, two cold repetitions and
512 committed decode tokens) is:

| Actual input | Prefill tok/s | Committed decode tok/s | Wall decode tok/s | Acceptance |
|---:|---:|---:|---:|---:|
| 3072 | 2415.00 ± 13.42 | 98.75 ± 0.02 | 98.59 ± 0.02 | 80.04% |
| 8192 | 2445.95 ± 5.48 | 100.16 ± 0.01 | 100.07 ± 0.02 | 80.44% |
| 16384 | 2404.93 ± 4.30 | 95.48 ± 0.01 | 95.31 ± 0.02 | 77.27% |
| 32768 | 2298.05 ± 0.04 | 89.15 ± 0.01 | 89.02 ± 0.01 | 74.68% |
| 65536 | 2089.18 ± 1.61 | 85.05 ± 0.11 | 84.99 ± 0.11 | 80.22% |
| 85000 | 1972.43 ± 2.89 | 81.53 ± 0.09 | 81.47 ± 0.09 | 80.58% |

The native FP8 capacity sweep (512-token input, 256 committed outputs, chunk 1024) measured:

| Capacity | Prefill tok/s | Decode phase tok/s | Decode wall tok/s |
|---:|---:|---:|---:|
| 1024 | 2003.40 ± 38.05 | 101.52 ± 0.06 | 101.49 ± 0.07 |
| 2048 | 1996.61 ± 50.16 | 101.43 ± 0.08 | 101.40 ± 0.08 |
| 4096 | 1997.52 ± 47.56 | 101.48 ± 0.01 | 101.45 ± 0.01 |
| 8192 | 1986.08 ± 56.70 | 101.12 ± 0.05 | 101.10 ± 0.05 |
| 16384 | 1990.54 ± 54.50 | 101.55 ± 0.01 | 101.52 ± 0.01 |
| 32768 | 2000.66 ± 49.85 | 101.33 ± 0.20 | 101.29 ± 0.20 |
| 65536 | 1999.82 ± 41.27 | 101.26 ± 0.06 | 101.20 ± 0.06 |

All allocations, output IDs and speculative counters repeated exactly. Plain FP8 controls
were 40.98 tok/s at 3072 and 33.12 tok/s at 85000. The FP8 teacher-force gate records three
speculative argmax differences (maximum emitted-logit deficit 0.125), while ordinary cold
re-prefill records one at the same 0.125 bound; this is the documented numerical envelope,
not a claim of token-by-token losslessness.

The same four-card host has passed the production four-rank collective suite and the BF16
quarter-shard projection/residual suite (10 geometry/phase cases, eager plus graph replay,
zero failures). Four-rank NVFP4/row-FP8/native block-FP8 attention and GDN projections, including
quarter-shard BF16 GDN control at T=1/4/128/2560, pass independent FP64 checks. A short four-card
NVFP4 ordinary eager request generates coherent code; this is a smoke check, not a throughput
or quality score. Local SM70 tests pass
the native block-FP8 Linear, fused SwiGLU/residual FP64 oracles, exact TP4 tensor slicing,
GQA 6Q/1KV and GDN 4QK/12V replay. These qualify operators, not a full four-card request.

The Q4 optimized proposal-head quarter shard is now registered and passes its FP64 operator
checks. Parallel proposal selection merges local FP32 winners/I32 IDs instead of gathering the
whole draft vocabulary; target logits remain complete for sampling. Local TP2 MTP graph/eager,
sampled routing, B=2 state and all 64 teacher-forced output positions pass after this change,
with zero argmax disagreements. TP4 graph residency has its own startup budget, now verified
by the four-card NVFP4 MTP startup/state gate.

The native artifact is converted from the friend's existing OneCat FP8 source, preserving
E4M3 codes and every 128×128 BF16 multiplier; no requantization is performed. BF16 embeddings,
output head and MTP stem remain BF16. The result is 31.30 GB (decimal). Its identity is
`qwen3.8-27b/fp8`, separate from the NVFP4 package's row-scaled FP8 projections.
See the [native artifact contract](docs/maintainer/qwen3.8-27b-artifact.md#native-block-128-fp8-textmtp-package).

The selected TP4 domain is Qwen3.8-27B NVFP4/native FP8 Text/MTP on SM70. Native FP8 requires
TP4; neither TP4 route admits Vision or DFlash. Its graph/eager, MTP commit, prefix-cache,
HTTP and occupancy/capacity measurements above were obtained on the separate four-card host.
The existing TP2 and 1Cat tables above are not relabeled as TP4 measurements.

### P2P enabled

Measured on 2026-10-01 with two V100-SXM2 16 GB cards at 300 W each, CUDA 12.8, PCIe 3.0 ×16,
PHB topology, `iommu=pt`, and identity IOMMU domains. NVIDIA P2P read/write checks pass and
NInfer automatically enables direct P2P. No inference algorithm or weight changes were needed
for the MTP P2P A/B evaluation. Both transport areas use the same launch commands; startup
qualifies the actual bidirectional copy route. `iommu=pt` alone does not prove working P2P.

#### QUASAR NVFP4

The separate `qwen3.8-27b/quasar-nvfp4` profile now loads the published
[QUASAR NInfer v3 artifact](https://huggingface.co/MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer)
through the public Engine. It admits SM70 TP2 Text/None/MTP/DFlash and is the recommended text
model; the convenience launcher's default is unchanged. Source codes/scales and all 256
activation-divisor pairs are retained. QUASAR is QAT, not mathematically lossless compression,
and the conversion includes BF16/W8 boundaries.
See the [artifact contract](docs/maintainer/qwen3.8-27b-artifact.md#13-quasar-trial-artifact).

Attention/GDN input and output weights now join MLP in the load-time Volta QPN layout; narrow
inputs write final output planes directly. Wide input projections decode weights once per call
for CUTLASS, then distribute the BF16 result. Stored weights/divisors are unchanged; no A4
activation quantization or attention-history pruning is enabled.

2026-10-03 local two-V100 measurement: PCIe P2P **without NVLink**, 300 W/card, CUDA 12.8,
saved code inputs, chunk 2560, complete INT8 group-64 KV, greedy optimized-head MTP3 and
CUDA Graphs. Two measured requests per row, no prefix reuse; 3072-input rows have no warmup,
the 85000-input row follows one complete warmup. Each emits one prefill token plus **512
committed decode tokens**. Decode below divides committed output by request wall time after
the first token; model loading and graph priming are excluded. Values are mean ± sample SD.

| Route | Actual input / capacity | Prefill tok/s | Committed wall decode tok/s | Acceptance |
|---|---:|---:|---:|---:|
| Before QPN input integration | 3072 / 8192 | 376.71 ± 0.77 | 102.02 ± 0.06 | 85.78% |
| QPN / MTP3 | 3072 / 8192 | 1751.55 ± 30.90 | 139.80 ± 0.08 | 85.19% |
| QPN / MTP3, warmed | 85000 / 180000 | 1315.28 ± 2.61 | 100.34 ± 0.05 | 86.62% |

The paired short-input gain is **4.65× prefill / 1.37× decode** despite slightly lower MTP
acceptance. Warmed 85K mean prefill/decode-phase/total request times are 64.625/5.100/69.729 s;
its Engine decode-phase rate is 100.40 tok/s. MTP weights remain **8.66 GiB/card**.
Both repetitions reproduce outputs and speculative counters within each route. Before/after
short-input outputs first differ at output position 270: numerical route changes are not
bit-identical trajectory preservation or proof of no quality loss. Independent FP64 input
projection checks cover real TP1/TP2 shapes and QPN/CUTLASS boundaries; the public Engine MTP
and DFlash gates each pass 64 strict teacher-forced positions and graph/eager sampling checks.
No BF16 quality score or full-180K/256K occupied-context result is claimed.
Reports: `profiles/bench/quasar-qpn-projections/`; initial pre-optimization comparisons remain in
`profiles/bench/quasar-v3-tp2-comparison/`.

QUASAR **DFlash7 JSONL**, on the same two cards: 98304 capacity, chunk 1024, complete INT8 KV,
full proposal head, greedy CUDA Graphs, no prefix reuse. These reuse the saved 32-record task
and distinct technical-document backgrounds; each row has one full warmup and two measured
requests. Model-default stopping is enabled. All six answers contain 1190 published tokens
(1189 timed after the first), stop naturally, and pass exact records, arithmetic and field-order
checks. Output IDs and draft counters repeat within each row.

| Actual JSONL input | Prefill tok/s | Committed wall decode tok/s | DFlash acceptance |
|---:|---:|---:|---:|
| 118 (native) | 564.22 ± 0.33 | 249.00 ± 0.08 | 99.05% |
| 4096 | 1577.84 ± 1.11 | 235.97 ± 0.04 | 99.05% |
| 85000 | 1184.19 ± 1.07 | 153.12 ± 0.02 | 98.30% |

85K JSONL mean prefill/decode-wall/total request times are **71.779/7.765/79.546 s**. This is
high-acceptance structured output, not a 249 tok/s claim for general code or prose; the MTP
code table above is a different workload, not a matched DFlash speedup comparison. Full-history
attention is retained. Report: `profiles/bench/quasar-qpn-projections/jsonl-dflash7.json`;
reproduction and exact answer checks: [stop-aware task benchmark](bench/README.md#stop-aware-v100-task-probes).

The completed **QUASAR task matrix** uses that same saved corpus, 98304 capacity, chunk 1024,
complete INT8 KV and one full warmup per case: MTP3 optimized head versus DFlash7 full head.
All 80 measured requests stop naturally below the 2048-output budget, without prefix reuse.
**D/M = DFlash7 / MTP3 committed wall decode tok/s**. Native rows average two repetitions;
every other task/length/backend has one measured request, not a stable mean.

| Actual input | Chinese story D/M | Translation D/M | 32-record JSONL D/M | Logic D/M |
|---:|---:|---:|---:|---:|
| Native: 129 / 395 / 118 / 417 | 56.45 / 89.58 | 162.90 / 143.66 | 247.73 / 163.33 | 204.68 / 152.93 |
| 1024 | 59.98 / 89.34 | 154.45 / 143.26 | 242.15 / 161.98 | 185.00 / 143.35 |
| 2048 | 58.38 / 85.06 | 149.46 / 140.75 | 232.84 / 158.92 | 193.55 / 151.54 |
| 4096 | 53.75 / 83.76 | 144.89 / 138.87 | 234.86 / 158.91 | 151.70 / 137.03 |
| 8192 | 55.62 / 82.65 | 145.07 / 139.49 | 229.81 / 156.56 | 195.49 / 143.77 |
| 16384 | 58.74 / 85.15 | 124.65 / 129.88 | 221.85 / 151.38 | 177.22 / 139.75 |
| 32768 | 48.32 / 77.06 | 124.26 / 124.26 | 198.65 / 139.45 | 162.39 / 132.71 |
| 65536 | 41.32 / 64.82 | 98.82 / 102.86 | 165.74 / 119.52 | 135.01 / 112.12 |
| 85000 | 39.26 / 61.28 | 87.06 / 93.29 | 152.72 / 111.73 | 127.43 / 105.94 |

At 85K, DFlash JSONL/logic decode gains are 36.69% / 20.28%, but story/translation are slower.
Its prefill is also slower: 1176.69–1186.82 versus MTP's 1226.43–1238.87 tok/s. JSONL's total
request changes only 79.895 → 79.794 s; logic is slower overall despite faster decode.
All JSONL exact-record/order and logic mapping/CHECK checks pass. Translation passes heading
and glossary checks, not an independent accuracy score. Stories satisfy the requested length
only at 8K/16K/64K. **31/36 paired outputs have identical IDs**, including all 85K pairs;
universal token parity or quality losslessness is not claimed.
The campaign resumed after unexpected host resets, keeping completed results. Its final runs
use at most about ten minutes active / three minutes idle; this does not prove continuous
hardware stability. [Full phase, acceptance, output-count and quality data](docs/performance.md#quasar-v3-stop-aware-task-matrix).

QUASAR already loads **zero Vision objects onto either GPU** and rejects Vision requests.
The published file is retained unchanged: its validation-only Vision payload occupies disk,
not the GPU weight arena. Removing that payload would not further reduce this text-only
profile's GPU memory or explain a throughput gain.

The downloaded model is `/Models/ninfer-V100X2/quasar-v3/qwen3_8_27b_nvfp4.ninfer`
(18.42 GiB; published SHA256 verified). This checkout's tested build is `build-v100-tp4`;
the following still runs **TP2**, not TP4:

```bash
LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64" \
  build-v100-tp4/apps/ninfer /Models/ninfer-V100X2/quasar-v3/qwen3_8_27b_nvfp4.ninfer \
  --tp 2 --devices 0,1 --max-context 8192 --kv-dtype int8 --prefill-chunk 2560 \
  --spec mtp --draft-tokens 3 --lm-head-draft --no-thinking --greedy \
  --max-new 512 --prompt 'Write a bounded blocking queue in C++.'
```

For DFlash, use `--max-context 98304 --prefill-chunk 1024 --spec dflash --draft-tokens 7`
and omit `--lm-head-draft` to match the full-head measurements above.

#### NVFP4 v3 occupied-context sweep

Official `qwen3.8-27b/nvfp4` v3 artifact, TP2, complete INT8 group-64 KV, greedy MTP3, optimized
draft head, CUDA Graphs, 3,072-token prefill chunks and **180,000 context capacity** (180032 KV
positions allocated). Each cell uses two cold-prompt requests without prefix reuse or an extra
request warmup; graphs are primed before measurement. Every request generates 513 tokens:
one from prefill and **512 timed committed decode tokens**. Rates are mean ± sample SD.
This recorded sweep predates the greedy/collective update below; its other occupancies have
not been remeasured with that update.

| Actual input tokens | Prefill tok/s | Committed decode tok/s | MTP acceptance |
|---:|---:|---:|---:|
| 3,072 | 1,783.96 ± 43.38 | 105.48 ± 0.01 | 72.97% |
| 8,192 | 1,754.43 ± 18.24 | 114.49 ± 0.01 | 81.98% |
| 16,384 | 1,699.06 ± 9.73 | 108.29 ± 0.003 | 79.47% |
| 32,768 | 1,596.24 ± 2.94 | 101.63 ± 0.03 | 79.69% |
| 65,536 | 1,397.83 ± 0.72 | 88.34 ± 0.04 | 79.12% |
| 85,000 | **1,306.48 ± 3.95** | **83.18 ± 0.13** | **79.12%** |

Both repetitions at every length produced identical output IDs and MTP statistics, without
EOS/EOG. Each prompt preserves the task and assistant suffix; its source body and acceptance
rate differ, so the 8K result does not imply that longer context is intrinsically faster. These
are fixed output windows, not complete-program quality scores.

At 85K, the same-input peer-off control below measured 79.18 tok/s: direct P2P improves committed
decode by **5.05%**, with all 513 output IDs and MTP statistics identical across both paths and
repetitions. The resident request averages 65.061 s prefill, 6.155 s decode and 71.220 s total;
model loading takes 19.662 s once, outside request timing. The historical 78.424 figure below
used the pre-reboot corpus and is not the causal P2P baseline.

#### Qualified NVFP4 prefill update

A matched 85K-input / 180K-capacity MTP3 comparison at chunk=2,560, with two cold requests
per implementation, measures **1,297.05 ± 2.21 → 1,311.92 ± 2.06 prefill tok/s (+1.15%)**
for the wider TP2 MLP tile. All 513 output IDs and acceptance counters match. Decode measures
81.63 → 81.71 tok/s; this does not establish a stable decode improvement. Chunk=2,560 was used
because desktop VRAM occupancy made larger chunks fail the startup allowance during this run.
The earlier chunk=3,072 table is not its matched control. Independent numerical criteria were
not relaxed; no new lossy attention option qualified for delivery.

#### Exact greedy and small-message P2P update

Same saved 85,000-token input, 180,000 capacity, chunk=2,560, NVFP4 v3, INT8 KV, greedy
optimized MTP3 and CUDA Graphs; two cold requests per implementation:

| Implementation | Prefill tok/s | Committed decode tok/s |
|---|---:|---:|
| Control, full-logit verification | 1,315.36 ± 2.45 | 81.80 ± 0.08 |
| Exact shard winners | 1,312.31 ± 3.15 | 82.40 ± 0.04 |
| Parallel shard argmax | 1,312.38 ± 1.68 | 82.43 ± 0.13 |
| Plus small-message direct-peer sums | 1,311.95 ± 3.70 | **85.21 ± 0.06** |

Combined decode gain is **4.17%**; the collective step contributes 3.37% over the preceding
route. The isolated argmax step is within end-to-end noise; no prefill gain is claimed.
Every run matches all 513 output IDs and acceptance fields: 357/463 accepted drafts over
155 rounds (77.11%). Weights, full attention history, KV precision and rounding boundaries
are unchanged. Sampling keeps full logits and its existing penalty updates.
The delivered resident request averages 64.789 s prefill, 6.009 s decode and 70.803 s total.
These paired results are not compared causally with the earlier chunk=3,072 sweep.
[Phase attribution and verification limits](docs/performance.md#exact-greedy-and-small-message-p2p-update).

#### NVFP4 v3 capacity sweep: fixed 512-token input

Only the maximum context changes. TP2, INT8 KV, greedy MTP3, optimized draft head, CUDA Graphs
and 1,024-token chunks stay fixed. Each capacity uses one discarded warmup and three measured
requests, with 512 input tokens and 256 timed decode tokens (257 total output tokens).
Decode here uses first-token-to-completion **wall time**, not just Engine decode-phase time.

| Maximum context | Wall decode tok/s | Prefill tok/s | MTP acceptance |
|---:|---:|---:|---:|
| 1,024 | 106.12 ± 0.01 | 1,443.78 | 70.04% |
| 2,048 | 106.12 ± 0.04 | 1,438.44 | 70.04% |
| 4,096 | 105.96 ± 0.02 | 1,440.81 | 70.04% |
| 8,192 | 105.98 ± 0.01 | 1,442.96 | 70.04% |
| 16,384 | 106.06 ± 0.04 | 1,443.37 | 70.04% |
| 32,768 | 106.00 ± 0.03 | 1,443.77 | 70.04% |
| 65,536 | 106.01 ± 0.06 | 1,443.42 | 70.04% |

All seven capacities were allocated exactly. All 21 measured runs produced the same 257 IDs
and acceptance statistics: 173/247 drafts accepted over 83 rounds per request. This is a
short-input capacity check, not a 64K-filled-context result.

#### Communication and regression checks

The 10 KiB BF16 all-reduce measured 26.6085 µs mean, 25.901 µs p50 and 43.910 µs p99 over
500 host-synchronized iterations. The matched peer-off path measured 47.7961 µs mean:
**44.33% lower communication latency**, not 44.33% end-to-end inference improvement. Both
paths passed exact-transfer, uneven-shape, guard and 64-consecutive-round checks.

Q4_K_M and NVFP4 v3 passed the real **text-only** TP2 MTP and prefix-cache regressions. Graph/eager outputs,
logits, acceptance and retained frontiers agree; each artifact had zero disagreements at
64 teacher-forcing positions. At a 3,274-token prompt, median cold/cached TTFT was
3.33492 s / 16.8689 ms for Q4_K_M and 3.01954 s / 14.6475 ms for NVFP4.
See [regression qualifications](docs/performance.md#p2p-enabled) for the existing Q4 cold/cache
near-tie differences and the limits of these checks.

#### DFlash2 v3 route (separate capacity)

The optional five-layer v3 drafter is measured separately from the 180K MTP acceptance profile.
The delivered route uses 98,304 capacity, 1,024-token chunks, full target INT8 KV,
TP2, greedy sampling, CUDA Graphs and DFlash7. It is text-only and needs the optional v3 drafter.
On the saved 85,000-token LRU-code input, one cold request per window measures:

| Timed committed tokens | Prefill tok/s | Decode tok/s | Accepted/drafted |
|---:|---:|---:|---:|
| 512 | 1,185.52 | **99.93** | 422/623 (67.74%) |
| 2,048, forced window | 1,182.63 | **95.87** | 1674/2612 (64.09%) |

The 512-token window has no EOS/EOG. The forced 2,048-token window continues past `<|im_end|>`
at total output token 1,150 because benchmark stopping is disabled; it is sustained execution
data, **not useful long-code completion speed**. All 513/2049 IDs and acceptance counters match
the preceding argmax-only route. The single-run sharded-selector differences (0.6–0.8%) do not
establish a stable speedup. Independent selector and real graph/eager / 64-position
teacher-forcing checks pass; these are not universal model-quality scores.

On the separate 85K high-code corpus, single forced 2,048-token windows measure DFlash3/5/7
at 79.30/76.10/87.91 tok/s versus MTP3 at 86.14, all at 98,304 capacity and chunk=1,024.
These windows continue past the first `<|im_end|>` at total output token 942 (917 for DFlash5),
so they are stress measurements, not useful completion speeds. DFlash3/7 match the MTP output
IDs; DFlash5 does not. DFlash7 also takes longer for the complete resident request
(95.40 versus 92.92 seconds). Fresh 1,024-token windows are slower; larger experimental drafts
9/11/15 brought no benefit and were removed. The supported 27B maximum remains seven.

Earlier P2P-enabled v3 snapshots, before these vocabulary-transfer reductions: 3,072 input / 512
timed output tokens measured 143.06 tok/s (two runs). A different 85K code corpus measured DFlash7
73.38 tok/s (47.18% acceptance) versus MTP3 83.13 tok/s (78.56%). Different corpora, windows and
acceptance prevent using those numbers or the peer-off MTP table as the current run's speedup
baseline. See [DFlash method and reproduction](docs/performance.md#dflash2-v3-separate-capacity).

#### Stop-aware non-code sweep

The same v3 artifact was tested with story, translation, 32-record JSONL and logic tasks:
**80 requests**, fixed 98,304 capacity / chunk 1,024 / TP2 / INT8 KV / greedy / CUDA Graphs,
no prefix reuse, and model-default stopping enabled. All requests finish at the first model
end token rather than continuing past EOS. D/M below means **DFlash7 / MTP3**, in committed
decode tok/s. Native prompt sizes are 129 / 395 / 118 / 417 tokens in column order; their
rates are two-run means. Every other cell is one measured request, not a repeated mean.

| Actual input tokens | Story D/M | Translation D/M | JSONL D/M | Logic D/M |
|---:|---:|---:|---:|---:|
| native | 49.20 / 74.83 | 134.68 / 123.00 | 210.50 / 137.09 | 158.88 / 126.10 |
| 1024 | 51.24 / 76.66 | 127.44 / 117.58 | 206.30 / 135.98 | 157.87 / 125.01 |
| 2048 | 51.09 / 74.41 | 128.45 / 116.14 | 200.78 / 133.63 | 173.32 / 126.34 |
| 4096 | 47.60 / 70.98 | 116.31 / 113.49 | 201.09 / 133.82 | 162.94 / 125.98 |
| 8192 | 49.11 / 70.17 | 125.74 / 116.00 | 198.69 / 132.06 | 166.37 / 125.39 |
| 16384 | 44.80 / 68.16 | 112.08 / 110.15 | 192.60 / 128.33 | 161.27 / 120.91 |
| 32768 | 41.44 / 63.23 | 110.56 / 108.44 | 175.01 / 119.59 | 142.03 / 111.83 |
| 65536 | 36.02 / 55.63 | 85.29 / 89.32 | 148.14 / 104.67 | 122.20 / 98.14 |
| 85000 | 32.26 / 51.67 | 79.61 / 84.86 | 137.65 / 98.77 | 111.05 / 91.60 |

JSONL is faster with DFlash at every tested length; at 85K, acceptance is 98.30% and decode
is 39.37% faster with identical output IDs. Story acceptance is only about 12–14%, making
DFlash slower than MTP. DFlash prefill is slower throughout this sweep, so faster decode
does not imply a faster complete request: at 85K, logic takes 78.47 / 76.87 seconds D/M.

All JSONL exact checks and logic CHECKs pass; translations pass section/glossary checks only.
Some stories violate length/literal requirements, including a missing `ORCHID-37` in the
native DFlash story. Only 30 of 36 task/length pairs have identical output IDs, so this is
not a universal token-parity or no-quality-loss result. All four 85K pairs match exactly.
[Full prefill, acceptance, output-length and whole-request tables](docs/performance.md#stop-aware-non-code-workloads-dflash7-versus-mtp3)
include the per-row output qualifications.

### P2P disabled

This area covers verified CUDA-managed UVA D2D staging, not an explicit NInfer pinned-host
copy branch. Before `iommu=pt`, the host's translated `DMA-FQ` domains prevented direct P2P.
The current peer-off A/B control uses a process-local CUDA capability-query shim, without
changing system settings. It is a diagnostic, not an advertised CLI switch.

#### NVFP4 v3 same-input 85K control

The same 85,000 input IDs, 180,000 capacity, 512 timed decode tokens, INT8 KV, TP2, greedy MTP3,
optimized draft head and CUDA Graphs as the P2P-enabled run, with two repetitions:

| Prefill tok/s | Committed decode tok/s | Wall decode tok/s | MTP acceptance |
|---:|---:|---:|---:|
| 1,297.94 ± 2.66 | 79.18 ± 0.014 | 79.151 | 79.12% |

Both communication paths accepted 360/455 drafts over 152 rounds per request. All output IDs
and speculative fields match. The same-input improvement is 5.0459% decode / 0.6578% prefill.

#### Earlier official NVFP4 v3 artifact validation

`neroued/Qwen3.8-27B-nvfp4-NInfer/qwen3_8_27b_nvfp4.ninfer` is a 23.72-GB, 1246-physical-object
`NINFER\x00\x03` container. Its projected `qwen3.8-27b/nvfp4` identity loaded on both V100s with
INT8 KV and MTP3 at `--max-context 180000` (180032 allocated KV positions); 671 tensors and six
frontend resources were materialized per the normal Engine path. A short greedy MTP check produced
21 tokens in five rounds with 100% acceptance (120.66 committed tok/s at 1024-token capacity).
The short check is a compatibility smoke test, not a long-context performance claim. A stable
512-prompt/512-output benchmark measured 1,344.24 ± 33.15 prefill tok/s and 98.831 ± 0.027
committed decode tok/s. At the acceptance workload (85,000 occupied tokens, 180,000 capacity,
512 output tokens), the same v3 artifact measured 1,277.61 ± 3.12 prefill tok/s and 78.424 ± 0.0004
committed decode tok/s over two repetitions before P2P was enabled; MTP3 acceptance was 79.12%.
These earlier runs use the pre-reboot corpus, not the current same-input A/B control.
The 120.66 figure is therefore a real short-window peak, not the representative decode rate.

#### Earlier Q4_K_M prefill

TP2, INT8 KV, `prefill_chunk=4096`, and 180,000-token capacity:

| Occupied prompt | Prefill throughput | Measurement |
|---:|---:|---|
| 8,192 tokens | 1,672.9 tok/s | one cold run |
| 85,000 tokens | 1,251.44 ± 2.87 tok/s | two cold runs |

The 85K prompt exceeds the requested 1,000 tok/s target. On the 8K probe, chunk sizes
1,024/2,048/3,072/4,096/5,120/8,192 measured 1,454.4/1,599.3/1,636.1/1,672.9/1,672.3/1,195.2
tok/s. For the 85K corpus, chunks 1,024/2,048/4,096 measured 1,133.0/1,213.4/1,251.0 tok/s.
The 4,096-token chunk reserves 1.49 GiB per device. These are prefill measurements, not decode
rates.

The staging TP2 collective path was also run independently on the two V100s: a 10 KiB BF16
all-reduce (the decode-shaped hidden block) measured 48.16 µs mean, 47.09 µs p50 and 64.92 µs p99
over 500 host-synchronized iterations. The test passed all exact-value, guard, uneven-shape and
64-consecutive-round checks. Weight quantization cannot eliminate the fixed collective schedule;
this microbenchmark alone does not establish a hardware-only decode ceiling.

#### Earlier Q4_K_M decode

With exactly 85,000 occupied prompt tokens, 180,000 capacity, TP2, INT8 KV, MTP3, optimized draft
head and CUDA Graphs, the pre-P2P staging path measured two 128-token windows:

| Repetitions | Prefill rate | Committed decode rate |
|---:|---:|---:|
| 2 | **1,251.44 ± 2.87 tok/s** | **50.68 ± 0.04 tok/s** |

Both windows used the same 85K raw token corpus and produced no EOS/EOG; aggregate MTP acceptance
was 78.07%. A separate
512-token occupied prompt at the same capacity
measured **60.0291 ± 0.0571 tok/s** over three 256-token decode windows; this is not an 85K
occupied-context result.

The superseded explicit pinned-host experiment reported **53.4075 ± 0.0639 tok/s** over three
512-token windows; it is not the retained transport implementation and is historical context only.
An earlier matched diagnostic used LM Studio CUDA 2.33.0's automatic GPU split with the same
85,000 prompt IDs, source GGUF, greedy sampling, Q8 KV, maximum context 180,000 (backend rounded
to 180,224), and max-three/min-zero MTP. It measured 35.4977 tok/s in one run. LM Studio's
single-run result is a diagnostic, not a repeated comparison.
Earlier user-reported 45/57 tok/s figures lacked complete workload metadata and are not used as
measured acceptance results.

#### Earlier Q4_K_M maximum-context capacity sweep

This earlier Q4_K_M sweep changes only the configured maximum context. It uses a 512-token code prompt,
a 256-token decode window, greedy sampling, Q8/INT8 KV and MTP3. Each cell is the mean ± sample
standard deviation of three measured runs after one warmup; prompt occupancy is only 512 tokens.
NInfer uses TP2 and its optimized draft head. LM Studio CUDA 2.33.0 uses automatic two-GPU
splitting with maximum-three/minimum-zero drafts.

| Maximum context | NInfer decode tok/s | LM Studio decode tok/s |
|---:|---:|---:|
| 1,024 | 60.06 ± 0.02 | 62.67 ± 0.24 |
| 2,048 | 60.12 ± 0.02 | 62.78 ± 0.05 |
| 4,096 | 60.11 ± 0.03 | 62.69 ± 0.10 |
| 8,192 | 60.13 ± 0.06 | 62.78 ± 0.07 |
| 16,384 | 60.14 ± 0.04 | 62.63 ± 0.06 |
| 32,768 | 60.16 ± 0.05 | 62.58 ± 0.004 |
| 65,536 | 60.06 ± 0.05 | 62.49 ± 0.16 |

Both engines honored all seven capacities. The output IDs were identical within each engine and all
windows were EOS/EOG-free. With this short prompt neither engine slowed materially as the capacity
increased; LM Studio was about 4% faster. This table is a capacity-setting comparison, not an
85K-filled-context benchmark.

#### Earlier NVFP4 at 512-token input

The corrected NVFP4 implementation uses a 512-token code prompt and 256 timed decode tokens, with one
warmup and three measured requests per capacity. TP2, INT8 KV, greedy MTP3, optimized draft head,
CUDA Graphs and 1,024-token prefill chunks are fixed. Decode rates are mean ± sample standard deviation;
decode uses first-token-to-completion wall time.

| Maximum context | NVFP4 decode tok/s | Prefill tok/s | MTP acceptance |
|---:|---:|---:|---:|
| 8,192 | 97.65 ± 0.08 | 1,369.9 | 70.04% |
| 16,384 | 97.67 ± 0.09 | 1,370.9 | 70.04% |
| 32,768 | 97.76 ± 0.11 | 1,370.8 | 70.04% |
| 65,536 | 97.55 ± 0.15 | 1,369.9 | 70.04% |

All four requested capacities were allocated exactly. All 12 measured runs produced the same
257 output IDs, without EOS/EOG; each accepted 173/247 drafts over 83 rounds. These are short-input
capacity checks, not 8K–64K occupied-context decode. The LM Studio table above uses Q4_K_M;
it is not a matched NVFP4 comparison.

#### Earlier NVFP4 at 85K occupied context

Qwen3.8-27B NVFP4, exactly **85,000 prompt tokens**, **180,000 capacity**, TP2, complete INT8
group-64 KV, greedy sampling, MTP3, optimized draft head, CUDA Graphs and 3,072-token prefill chunks:

| Implementation | Runs | Prefill tok/s | Committed decode tok/s | MTP acceptance |
|---|---:|---:|---:|---:|
| FP32 SwiGLU, shared attention scores/reducer weights, prepared GDN Q/K | 2 | **1,279.44 ± 2.94** | **78.36 ± 0.04** | **79.12%** |

Each run generates 513 tokens: one from prefill and **512 timed decode tokens**. The optimized
runs produce identical IDs to each other, with 360/455 accepted drafts over 152 rounds per run.
These exceed the requested 1,000 prefill / 70 committed decode tok/s targets at 85K occupancy.
The 512-token output limit is a throughput window, not a completed-program quality evaluation.
The complete attention operator passes its independent FP64 oracle, including all 24 heads and
four queries at 85K with a 180K envelope. This verifies the measured numerical and generation
behavior; it is not a general coding-quality evaluation.
The GDN path also passes the independent FP64 recurrence oracle for its outputs and final state,
including 3,072 tokens with the real TP2 head geometry and nonzero initial state.

Earlier NVFP4 results, including **78.90 tok/s**, used a wide SwiGLU path that rounded gate/up to
BF16 before SiLU and multiplication. Expanded independent FP64 tests exposed excessive error in
both FP8 and NVFP4 routes. Keeping those intermediates in FP32 fixes the failing tests, including
both TP2 shards at 3,072 tokens, without widening tolerances. The generated sequence changes;
the old figures are real measurements but are **not the current numerical baseline**. Measurements
taken with the earlier mismatched FP8 output-head path are also excluded. A faster experiment
that reassociated the FP32 sum changed this corpus's generated sequence; the delivered reducer
preserves the original order and introduces no additional quantization or approximate attention.
The 3,072-token chunk reserves 1.40 GiB of workspace per device; 4,096 does not fit this NVFP4
artifact together with 180K capacity on the measured host.

This earlier staging request averages 1.57 ms of prompt preparation, 66.435 s of prefill and 6.534 s of
decode, or 72.974 s total. The same invocation loads the resident model once in 18.65 s, including
16.25 s of upload. These timings cover the prepared prompt path only.
The complete per-stage GPU breakdown, MTP verify/proposal costs, copy activity, timing gaps and
remaining optimization decisions are in [the staging performance ledger](docs/performance.md#nvfp4-full-request-ledger).
Current P2P reproduction commands are in [performance methodology](docs/performance.md#p2p-enabled).

An earlier NVFP4 TP2 comparison against the `plus1998/Ninfer-V100-Duo` code path at a 3K prompt,
98,304 capacity and MTP3 measured 976 tok/s prefill / 69.22 tok/s decode in this fork versus
973.5 / 68.96 tok/s upstream. This is a short-input cross-check only, not an 85K result. The
upstream repository and its reported numbers should not be treated as measurements of this fork.

#### Historical DFlash2 experiment

The old fixed-85K figures (DFlash3 **25.19 tok/s**, DFlash7 **20.52 tok/s**) used the pre-v3
container and old KV layout. They are retained only as historical context; the current v3 result
is the 98,304-capacity measurement above, and 180K DFlash is intentionally not advertised.

## Build and run

Requirements for this profile: 64-bit Linux, NVIDIA driver, CUDA 12.8, CMake 3.28+, C++20 host
compiler, Ninja, pkg-config, FFmpeg development libraries (`libavformat >= 60`, `libavcodec >= 60`,
`libavutil >= 58`, `libswscale >= 7`) and libcurl >= 7.85. Build dependencies locally when the
system versions do not meet those requirements:

```bash
tools/v100/build_dependencies.sh
PKG_CONFIG_PATH="$PWD/build/_deps/install/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" \
cmake -S . -B build-v100 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build-v100 -j2
```

For the recommended QUASAR model, download the published v3 container with an existing Hugging
Face CLI and verify the publisher's checksum (no local weight conversion is needed):

```bash
hf download MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer \
  qwen3_8_27b_nvfp4.ninfer SHA256SUMS \
  --revision v3 --local-dir /Models/ninfer-V100X2/quasar-v3
(cd /Models/ninfer-V100X2/quasar-v3 && sha256sum --check SHA256SUMS)
```

Run QUASAR with TP2, 180K capacity, 2560-token prefill chunks, INT8 KV and optimized-head MTP3:

```bash
NINFER_V100X2_ARTIFACT=/Models/ninfer-V100X2/quasar-v3/qwen3_8_27b_nvfp4.ninfer \
NINFER_V100X2_PREFILL_CHUNK=2560 \
tools/v100/ninfer-v100x2.sh \
  --prompt "Explain prefill and decode in three sentences." \
  --max-new 128 --greedy --no-thinking
```

Start its OpenAI/Anthropic API with the same MTP3 profile:

```bash
env LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  build-v100/apps/ninfer-serve /Models/ninfer-V100X2/quasar-v3/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 8080 \
  --tp 2 --devices 0,1 --max-concurrency 1 \
  --max-context 180000 --kv-capacity 180000 --kv-dtype int8 --prefill-chunk 2560 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --default-max-tokens 65536 --no-thinking
```

For DFlash7, set both capacities to `98304`, the chunk to `1024`, and replace the speculative
options with `--spec dflash --draft-tokens 7`, omitting `--lm-head-draft` to match the full-head
measurements. Output length is bounded by remaining context. See [serving](docs/serving.md)
for requests and streaming. Neither QUASAR profile supports Vision.

For the alternative Q4_K_M profile, convert the local LM Studio source once (Python 3.11 with NumPy):

```bash
python3 -m tools.convert.qwen3_8_27b.convert_gguf \
  --model /Models/LM-Studio-models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf \
  --mmproj /Models/LM-Studio-models/lmstudio-community/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-BF16.gguf \
  --out /Models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer
```

Run Q4_K_M with the unchanged launcher defaults (devices `0,1`, 180K capacity, 4K prefill chunks, INT8 KV,
MTP3 and optimized draft head):

```bash
tools/v100/ninfer-v100x2.sh \
  --prompt "Explain prefill and decode in three sentences." \
  --max-new 128 --greedy --no-thinking
```

Set `NINFER_V100X2_ARTIFACT`, `NINFER_V100X2_DEVICES`, `NINFER_V100X2_MAX_CONTEXT`,
`NINFER_V100X2_PREFILL_CHUNK`, `NINFER_V100X2_KV_DTYPE`, or
`NINFER_V100X2_DRAFT_TOKENS` to override the launcher defaults. The launcher does not constrain
host CPU affinity; benchmark CPU use should remain below the operator's 85% ceiling.

The current P2P corpus is local at `profiles/bench/v100-code-85000-iommu-pt.ids`; earlier staging
measurements used `/tmp/v100-code-85000.ids`. Corpora, model artifacts and raw profiler reports
are not included in the repository. See [performance methodology](docs/performance.md) for
corpus generation, benchmark commands and additional qualifications.

### Experimental TP2 Vision

Use the official NVFP4 artifact, not QUASAR or GGUF-derived Q4_K_M. Start with a small text/KV
capacity and visual-token budget on the two 16 GB cards; the command below is not a measured
maximum-capacity profile. `--vision-max-tokens` bounds merged visual tokens, while the complete
text-plus-media prompt must also fit `--max-context`.

```bash
env LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  build-v100/apps/ninfer-serve /Models/ninfer-V100X2/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 8080 --tp 2 --devices 0,1 --max-concurrency 1 \
  --max-context 4096 --kv-capacity 4096 --kv-dtype int8 --prefill-chunk 1024 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --vision --vision-max-tokens 1024 --no-prefix-reuse --default-max-tokens 512 --no-thinking
```

This disables Engine prefix reuse for all requests on this server, including text. It does not
disable the separate immutable-media preprocessing cache. The normal QUASAR text profile above
keeps prefix reuse. See [multimodal requests](docs/serving.md#multimodal-request) and
[known Vision test limitation](tests/README.md#experimental-tp2-vision-status).

## Verification

The CUDA 12.8 `sm_70` build was completed with `-j4`. The all-reduce integration test passed,
including exact cross-device byte probes and the 500-iteration 10 KiB microbenchmark. The focused
`nvfp4_a16`, `fp8_a16`, `swiglu_nvfp4`, and `swiglu_fp8` suites passed. Both SwiGLU suites check
full and TP2 shard outputs directly against the same FP64 mathematical oracle, including wide
prefill and caller-workspace sizing. FP8 A8 and NVFP4 A4 suites intentionally
report unsupported on V100 because those tensor-core routes require newer architectures; they are
not part of the V100X2 execution profile.

The decode update also passes `ninfer_gqa_attention_test` and `ninfer_attention_headlocal_test`,
covering BF16/INT8 caches, TP1/TP2, masks, fragmented pages, cache mutation and the full 85K FP64
oracle. FP8 regression cases cover native and prepacked full/TP2 vocabulary heads, single-token
calls and chunk tails. Both FP8 and NVFP4 prefill tests check native/prepacked real matrix shapes
at T=128/1024/4096 against the original represented weights. Q5 LinearAdd passes the FP64 oracle
at both TP2 row-shard extents, including its interior workspace maximum; BF16 Linear passes at
both row and column shards. The two-card LinearAdd and SwiGLU composition suites pass as well.
CLI, server and benchmark are rebuilt with these changes.

## Acknowledgements and license

V100X2 implementation work builds on [Neroued/ninfer](https://github.com/Neroued/ninfer), the
RTX 3060 TP2 fork, and [geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100).
Volta NVFP4 and TP2 behavior was cross-checked against
[plus1998/Ninfer-V100-Duo](https://github.com/plus1998/Ninfer-V100-Duo). The prefill investigation
consulted [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM). DFlash2 follows
[Inco AI's implementation](https://inco.ai/blog/dflash2/) and its
[Qwen3.8-27B draft checkpoint](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2).
[KVMem](https://github.com/kvmem/kvmem-llama.cpp) informed the optional RAM-KV experiment.
The native prototype uses lexical retrieval, not KVMem's Q/K-vector retrieval; selective-history
attention is approximate and explicitly opt-in. The default profile retains full-context attention.

[Li3age](https://github.com/Li3age) contributed TP2 Vision and the configurable visual-token
budget in [PR #1](https://github.com/tuxKOH/ninfer-V100X2/pull/1).

The measured models derive from [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B).
The NVFP4 artifact uses the mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4), packaged by
[Neroued](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer).
The recommended QUASAR profile uses [QUASAR-QAT's checkpoint](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4)
and [MirkoCovizzi's NInfer conversion](https://huggingface.co/MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer).

NInfer and this fork are licensed under [Apache-2.0](LICENSE). See [NOTICE](NOTICE) for required
attribution; third-party dependencies retain their own license files.
