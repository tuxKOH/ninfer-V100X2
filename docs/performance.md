# QUASAR NVFP4 performance

Local hardware: 2 × Tesla V100-SXM2 16 GB, SXM2-to-PCIe adapters, PCIe 3.0 ×16 P2P without NVLink, CUDA 12.8 / SM70. Decode rates count committed tokens, excluding rejected drafts. Context capacity and occupied input length are reported separately. Loading and CUDA Graph priming are outside request timing.

[llama.cpp / LM Studio controls](llama-comparison.md) · [Other V100 profiles](performance-v100.md) · [TP4/NVLink](tp4.md) · [RAM-KV/PP2](experiments.md) · [Inherited upstream campaigns](performance-upstream.md)

## QUASAR v3: final long-context decode evaluation

Measured on 2026-10-13 on two V100-SXM2-16GB cards, CUDA 12.8 / SM70, qualified
PCIe P2P without NVLink. The artifact is `qwen3.8-27b/quasar-nvfp4`; the code input
is the saved `profiles/bench/v100-code-85000-iommu-pt.ids`. Both routes use 85000
actual input tokens, 180000 capacity (180032 allocated), complete INT8 group-64 KV,
chunk 2560, greedy optimized-head MTP4 and CUDA Graphs. Each has two measured cold
requests, no prefix reuse or extra request warmup, one prefill output and 512 timed
committed decode outputs. Loading and graph priming are outside request timing.
Rates are mean ± sample SD; wall decode is `512 / (total_seconds - first_token_seconds)`.

| Route | Prefill tok/s | Committed wall decode tok/s | MTP acceptance |
|---|---:|---:|---:|
| Default complete-history MTP4 | 1313.37 ± 3.89 | 126.894 ± 0.028 | 79.02% |
| `--volta-long-attention` / MTP4 | 1307.41 ± 2.96 | 143.459 ± 0.013 | 76.74% |

| Route | Prepare, s | Prefill, s | First token, s | Decode phase, s | Total request, s |
|---|---:|---:|---:|---:|---:|
| Default | 0.001544 | 64.719 | 64.721 | 4.031 | 68.756 |
| Experimental attention | 0.001593 | 65.014 | 65.016 | 3.566 | 68.585 |

Prepare uses saved IDs, not raw-text tokenizer latency. First-token time includes
prefill; the phase columns are not all additive. Experimental attention improves
decode 13.05% in this pair; prefill variation leaves total request time almost equal.
Both routes repeat all 513 output IDs and speculative counters and match their own
previously qualified controls. Their trajectories differ across profiles. The
requested 150 tok/s was not reached; the user accepted this result and ended that goal.

The retained default optimizations share QK/softmax work across dimension warps,
decode packed INT8 KV cooperatively, stage split-output reduction while retaining
its ascending FP32 association, and precast selected narrow NVFP4 activations.
Paired gate/up warp groups retain FP32 SiLU/product before one BF16 rounding.
The W8 vocabulary head has an exact load-time code/scale permutation, without
requantization or a second resident copy. Its narrow consumers, the W8 MTP
projections and the Q4 proposal head use measured unroll/direct-read paths.
Single-query attention uses eight-row Volta QPN tiles with the default eight-key
softmax and FP16 contribution rounding; five-query attention removes the shared
QK-score handoff while retaining its accumulation associations.

These routes have independent FP64 decoded-weight projection/attention oracles,
exact layout/codec checks, cache/guard checks and CUDA Graph replay evidence.
Twenty-five teacher-forced 85K vocabulary checkpoints of the default attention
changes equal the saved reference exactly. Real-model default and experimental
MTP4 gates separately check 64 graph/eager commits, acceptance counters, partial
commit/fallback and peer egress. Exactness is established for the represented
comparisons, not universal source-model quality.

The optional `--volta-long-attention` flag is startup-fixed and defaults off.
It supports SM70 27B TP2/PP1, one Text/None/MTP request and complete INT8 KV;
Vision, RAM-KV and DFlash are excluded. Above 16390 keys it selects FP32 PV
accumulation for one/five/six-query paths. One/six-query kernels use eight-row
QPN tiles and 32-key online-softmax updates; five-query verification combines
sixteen keys per update. Above 65536 keys those widths target about 1072
keys per split, reducing staging and final-reduction overhead. At 85K the
default has about 355 splits, versus 80 in the experimental profile. Every
causally visible historical KV entry remains included. Wide prefill is unchanged.

Full 85K attention passes the independent FP64 relative-L2 criterion of 0.00315,
with exact packed cache bytes and graph/eager boundary checks. However, the
experimental MTP4 profile fails the separate near-lossless full-vocabulary gate:
25/25 argmax matches, minimum cosine 0.998905, maximum KL 0.007842 nats, against
bounds of cosine ≥0.9999 and KL ≤0.005. More FP32 accumulation and changed
partition/softmax association do not imply bit identity or task-level equivalence.
This is why the profile remains opt-in.

Final local reports are `profiles/bench/quasar-150-tp2/final-20261013-mtp4-{default,long}.json`.
Reproduce either route with the public benchmark; add `--volta-long-attention`
for the experimental row:

```bash
LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64" \
  build-v100-tp4/bench/ninfer_bench \
  --weights /Models/ninfer-V100X2/quasar-v3/qwen3_8_27b_nvfp4.ninfer \
  --tp 2 --devices 0,1 --max-ctx 180000 --kv-dtype int8 --prefill-chunk 2560 \
  --spec mtp --draft-tokens 4 --lm-head-draft \
  --corpus profiles/bench/v100-code-85000-iommu-pt.ids \
  -pg 85000,512 --warmup 0 -r 2 --capture-generation -o json \
  --output-file profiles/bench/quasar-final-default.json
```

### Local GPQA-Diamond comparison: 104 paired questions

This is an **85K unrelated-code occupancy variant**, not standard short-context
GPQA-Diamond. A frozen data-independent shuffled order and identical question/
answer-choice prompts are shared by all four routes. Each input occupies 85000
tokens; capacity is 131072, output budget 32768, chunk 2560, greedy MTP4,
`reasoning_effort=xhigh`, complete INT8 KV and no prefix reuse. Embedded frontend
resources match across artifacts. Scoring uses the final `ANSWER: A/B/C/D`;
missing or truncated answers count as incorrect in the full score.

| Local artifact / profile | Correct / 104 | Accuracy | Output-budget truncations |
|---|---:|---:|---:|
| QUASAR, default attention | 88 | 84.62% | 11 |
| QUASAR, experimental attention | 90 | 86.54% | 8 |
| Official NVFP4 | 90 | 86.54% | 5 |
| GGUF-derived Q4_K_M | 87 | 83.65% | 5 |

Default → experimental has two losses and four gains, +1.92 percentage points;
paired McNemar exact two-sided p=0.6875. There is no detected accuracy decrease
in this sample, but no statistical noninferiority or universal quality-equivalence
claim follows. Truncation affects the score; no BF16/FP8 local reference was run.
The user accepted this comparison and stopped at 104 common questions. The
remaining 94 questions were not run; this is not a completed 198-question score.

Runner: `tools/v100/bench_gpqa.py` and the public-Engine `ninfer_v100_gpqa` target.
Saved local results: `profiles/bench/gpqa-85k-matrix/{default,long,nvfp4,q4,summary}.json`.
The summary preserves `complete=false`.

### Original short JSONL DFlash7 retest

The 2026-10-13 retest uses the original **32-record arithmetic JSONL generation**
case that measured 249.00 tok/s, not the demo-page JSON→JSONL conversion prompt.
Input is 118 tokens; capacity 98304, chunk 1024, complete INT8 group-64 KV,
greedy full-head DFlash7, CUDA Graphs, no prefix reuse and model-default stopping.
One full warmup precedes two measured requests.

| Measurement | Prefill tok/s | Committed wall decode tok/s | Acceptance |
|---|---:|---:|---:|
| Original 2026-10-03 | 564.22 ± 0.33 | 249.00 ± 0.08 | 99.05% |
| Final 2026-10-13 | 580.95 ± 0.01 | 255.31 ± 0.11 | 99.05% |

Each final answer publishes 1190 tokens, with 1189 timed after the first, and
stops naturally. All 32 record values, arithmetic and field order pass exact
checks. All output IDs and speculative counters equal the original measurement.
Committed throughput is 2.53% higher; **300 tok/s was not reached**. This is
high-acceptance structured generation, not a general code/prose speed claim.
The experimental attention flag is off and unsupported for DFlash.
Report: `profiles/bench/quasar-150-tp2/final-20261013-jsonl-native-dflash7.json`;
reproduce with `ninfer_v100_task_bench`, task `structured_jsonl`, occupancy `native`
and the existing `profiles/bench/other_inputs_20261001/cases.json`.
Detailed health logs accompany all three final cases: no reset, zero volatile
uncorrected ECC, maximum observed GPU temperature 64°C and host CPU use below 37%.
This short run does not establish long-duration hardware stability.

Release checks rebuild the CLI, server and affected benchmarks; artifact, CLI/server
option and report-contract tests pass. GPQA scoring tests and schema-v15 matrix
consumption pass, including both saved final reports. Current 85K MTP4 graph/eager
state checks pass with identical commit IDs/counters and zero peer-egress mismatches.
A broader public FP64 attention rerun was stopped by its wrapper's 560-second
timeout while computing the six-query reference, without a numerical failure being
reported; no completed verdict is claimed for that rerun. The affected five-query
operator and full-vocabulary comparisons above retain their completed evidence.

## QUASAR v3: Volta QPN projections

The separate `qwen3.8-27b/quasar-nvfp4` trial uses the published QUASAR NInfer v3 artifact
on this two-card PCIe-P2P host, without NVLink. QUASAR is QAT, not mathematically lossless
compression. Its stored NVFP4 codes/scales and FP32 activation divisors remain unchanged;
this does not establish BF16-equivalent artifact quality. The recommended profile is text-only:
without `--vision` it loads zero Vision objects onto either GPU, and validation-only Vision data
remains in the original file. `--vision` at TP2 binds that data on the primary device as an
experimental route.

SM70 Attention/GDN input and output projections now share the load-time QPN layout with MLP.
Narrow inputs write final output planes directly. Wide projections decode weights once per
call for CUTLASS and distribute the BF16 result. There is no second resident weight copy,
A4 activation quantization or attention-history pruning. Independent FP64 input-projection
oracles cover real TP1/TP2 geometries and QPN/CUTLASS boundaries. Public Engine MTP and DFlash
gates each pass 64 strict teacher-forced positions and graph/eager sampling checks.

Measured on 2026-10-03: two V100-SXM2 16 GB, 300 W/card, CUDA 12.8 / SM70, saved code inputs,
chunk 2560, complete INT8 group-64 KV, greedy optimized-head MTP3, CUDA Graphs and no prefix
reuse. Two measured requests per row; 3072-input rows have no extra warmup, and 85000 input
follows one complete warmup. Each generates one prefill output and 512 timed committed decode
outputs. Load and graph priming are outside timing. Decode is committed output divided by
wall time after the first token; rates are mean ± sample standard deviation.

| Route | Actual input / capacity | Prefill tok/s | Committed wall decode tok/s | Acceptance |
|---|---:|---:|---:|---:|
| Before QPN input integration | 3072 / 8192 | 376.71 ± 0.77 | 102.02 ± 0.06 | 85.78% |
| QPN / MTP3 | 3072 / 8192 | 1751.55 ± 30.90 | 139.80 ± 0.08 | 85.19% |
| QPN / MTP3, warmed | 85000 / 180000 | 1315.28 ± 2.61 | 100.34 ± 0.05 | 86.62% |

The matched short-input gain is 4.65× prefill / 1.37× decode. Warmed 85K mean prefill,
decode phase and total request times are 64.625 / 5.100 / 69.729 seconds; the decode-phase
rate is 100.40 tok/s. MTP weights occupy 8.66 GiB per card. Within each route, repetitions
reproduce all output IDs and speculative counters. Before/after short-input trajectories
first differ at output position 270; numerical qualification does not prove unchanged
end-to-end quality. No BF16 quality score or full-180K/256K occupied-context result is claimed.
Reports: `profiles/bench/quasar-qpn-projections/{before-mtp3,after-mtp3,after-85k-mtp3-warm}.json`.

## QUASAR v3: stop-aware task matrix

Measured on 2026-10-03 through the public Engine using the same frozen token IDs as the
official-v3 non-code sweep below. Four tasks cover a three-chapter Chinese story requesting
900–1200 Chinese characters and fixed literal items; six-section English-to-Chinese translation
with at least 20 glossary entries; exactly 32 prescribed JSONL records; and a five-slot logic
puzzle with a complete service/colour/port mapping and final `CHECK=4606`.

Both backends use TP2 devices 0,1, **98304 context/KV capacity**, chunk 1024, complete INT8
group-64 KV, greedy CUDA Graphs, zero presence/frequency penalties, disabled thinking/prefix
reuse and zero Vision GPU objects. MTP uses three drafts and the optimized proposal head;
DFlash uses seven drafts and the full proposal head. Every case has one full warmup; native
inputs have two measured repetitions per backend, all other points only one. All 80 measured
requests stop naturally at their first model end token before the 2048-output budget. Load,
graph priming, warmup and cooldown are excluded. Committed wall decode is
`(published_tokens - 1) / (total_seconds - first_token_seconds)`; phase decode is retained
separately in the report. Counts include the prefill output and terminating model token.

The actual input counts are 129 / 395 / 118 / 417 for native story / translation / JSONL /
logic, then exactly 1024 / 2048 / 4096 / 8192 / 16384 / 32768 / 65536 / 85000 for each task.
Distinct technical-document backgrounds precede the unchanged final task/template. Visible
background varies with length, so this is not a single-variable length or retrieval-quality
experiment. **D/M means DFlash7 / MTP3** in every paired table. Native decode shows mean ±
sample standard deviation; other entries are single measurements, not stable means.

### Chinese story

| Actual input tokens | Prefill tok/s D/M | Committed wall decode tok/s D/M | Acceptance D/M | Total output tokens D/M | Identical IDs |
|---:|---:|---:|---:|---:|:---:|
| 129 (native) | 631.17 / 632.74 | 56.45 ± 0.002 / 89.58 ± 0.007 | 11.43% / 39.80% | 990 / 1152 | No |
| 1024 | 1587.24 / 1676.19 | 59.98 / 89.34 | 13.73% / 40.11% | 1202 / 1202 | Yes |
| 2048 | 1588.60 / 1679.56 | 58.38 / 85.06 | 13.80% / 37.80% | 1034 / 1034 | Yes |
| 4096 | 1576.10 / 1666.56 | 53.75 / 83.76 | 11.42% / 36.47% | 934 / 934 | Yes |
| 8192 | 1548.49 / 1638.20 | 55.62 / 82.65 | 12.90% / 37.00% | 940 / 940 | Yes |
| 16384 | 1500.94 / 1583.26 | 58.74 / 85.15 | 15.47% / 41.59% | 952 / 952 | Yes |
| 32768 | 1415.61 / 1481.05 | 48.32 / 77.06 | 13.08% / 40.12% | 726 / 726 | Yes |
| 65536 | 1257.29 / 1322.89 | 41.32 / 64.82 | 13.72% / 38.96% | 796 / 796 | Yes |
| 85000 | 1176.69 / 1226.43 | 39.26 / 61.28 | 14.63% / 39.70% | 1077 / 1077 | Yes |

### English → Chinese translation

| Actual input tokens | Prefill tok/s D/M | Committed wall decode tok/s D/M | Acceptance D/M | Total output tokens D/M | Identical IDs |
|---:|---:|---:|---:|---:|:---:|
| 395 (native) | 1201.31 / 1238.48 | 162.90 ± 0.021 / 143.66 ± 0.044 | 60.36% / 83.74% | 998 / 858 | No |
| 1024 | 1587.78 / 1682.12 | 154.45 / 143.26 | 57.56% / 84.42% | 1027 / 1027 | Yes |
| 2048 | 1581.97 / 1678.69 | 149.46 / 140.75 | 57.28% / 84.32% | 1012 / 1012 | Yes |
| 4096 | 1570.32 / 1665.85 | 144.89 / 138.87 | 55.06% / 82.24% | 899 / 899 | Yes |
| 8192 | 1545.72 / 1637.66 | 145.07 / 139.49 | 56.61% / 85.52% | 924 / 975 | No |
| 16384 | 1505.51 / 1582.73 | 124.65 / 129.88 | 48.96% / 80.69% | 757 / 757 | Yes |
| 32768 | 1413.89 / 1480.07 | 124.26 / 124.26 | 56.03% / 85.08% | 882 / 882 | Yes |
| 65536 | 1264.06 / 1322.04 | 98.82 / 102.86 | 52.81% / 81.32% | 883 / 883 | Yes |
| 85000 | 1184.80 / 1232.95 | 87.06 / 93.29 | 49.87% / 78.03% | 723 / 723 | Yes |

### 32-record JSONL

| Actual input tokens | Prefill tok/s D/M | Committed wall decode tok/s D/M | Acceptance D/M | Total output tokens D/M | Identical IDs |
|---:|---:|---:|---:|---:|:---:|
| 118 (native) | 563.48 / 567.46 | 247.73 ± 0.134 / 163.33 ± 0.032 | 99.05% / 99.89% | 1190 / 1190 | Yes |
| 1024 | 1588.03 / 1680.68 | 242.15 / 161.98 | 99.05% / 100.00% | 1190 / 1190 | Yes |
| 2048 | 1581.29 / 1679.06 | 232.84 / 158.92 | 98.30% / 99.78% | 1190 / 1190 | Yes |
| 4096 | 1574.58 / 1665.46 | 234.86 / 158.91 | 99.05% / 100.00% | 1190 / 1190 | Yes |
| 8192 | 1549.08 / 1635.96 | 229.81 / 156.56 | 98.30% / 99.78% | 1190 / 1190 | Yes |
| 16384 | 1504.45 / 1581.44 | 221.85 / 151.38 | 98.30% / 99.89% | 1190 / 1190 | Yes |
| 32768 | 1413.90 / 1472.01 | 198.65 / 139.45 | 98.30% / 99.78% | 1190 / 1190 | Yes |
| 65536 | 1261.11 / 1318.70 | 165.74 / 119.52 | 98.30% / 99.78% | 1190 / 1190 | Yes |
| 85000 | 1180.44 / 1227.40 | 152.72 / 111.73 | 98.30% / 99.78% | 1190 / 1190 | Yes |

### Five-slot logic puzzle

| Actual input tokens | Prefill tok/s D/M | Committed wall decode tok/s D/M | Acceptance D/M | Total output tokens D/M | Identical IDs |
|---:|---:|---:|---:|---:|:---:|
| 417 (native) | 1240.28 / 1283.12 | 204.68 ± 0.034 / 152.93 ± 0.042 | 80.06% / 91.83% | 1301 / 1301 | Yes |
| 1024 | 1594.90 / 1681.08 | 185.00 / 143.35 | 71.69% / 84.41% | 656 / 656 | Yes |
| 2048 | 1581.55 / 1678.54 | 193.55 / 151.54 | 78.17% / 92.79% | 699 / 699 | Yes |
| 4096 | 1569.54 / 1664.90 | 151.70 / 137.03 | 58.27% / 81.33% | 514 / 514 | Yes |
| 8192 | 1547.80 / 1636.89 | 195.49 / 143.77 | 81.29% / 89.14% | 670 / 642 | No |
| 16384 | 1502.74 / 1580.28 | 177.22 / 139.75 | 75.55% / 89.77% | 699 / 697 | No |
| 32768 | 1412.87 / 1476.17 | 162.39 / 132.71 | 77.59% / 93.58% | 612 / 612 | Yes |
| 65536 | 1258.20 / 1316.37 | 135.01 / 112.12 | 77.25% / 91.89% | 693 / 693 | Yes |
| 85000 | 1186.82 / 1238.87 | 127.43 / 105.94 | 79.14% / 93.06% | 655 / 655 | Yes |

### Complete requests and output qualifications

| Task at 85000 input tokens | Prefill seconds D/M | Decode wall seconds D/M | Total resident request seconds D/M |
|---|---:|---:|---:|
| Chinese story | 72.237 / 69.307 | 27.404 / 17.560 | 99.642 / 86.868 |
| English → Chinese translation | 71.742 / 68.940 | 8.293 / 7.740 | 80.036 / 76.681 |
| 32-record JSONL | 72.007 / 69.252 | 7.786 / 10.642 | 79.794 / 79.895 |
| Five-slot logic puzzle | 71.620 / 68.611 | 5.132 / 6.173 | 76.754 / 74.785 |

At 85K, DFlash improves JSONL/logic decode by 36.69% / 20.28%, with identical complete output
IDs; story/translation decode is slower. DFlash prefill is slower throughout the matrix.
JSONL total request time differs by only 0.101 seconds (about 0.13%), within what a single
pair can establish; this is not claimed as a stable whole-request gain. Logic is slower
overall despite faster decode. The 247.73 tok/s native JSONL rate is high-acceptance structured
generation, not ordinary translation, general code or a 12K-input JSONL translation speed.

All ten JSONL completions per backend pass exact count/value/arithmetic/field-order checks.
All ten logic answers per backend contain the complete expected mapping and `CHECK=4606`,
including separate assignment lists and combined mappings. Translations retain six headings
and at least twenty glossary entries; this is not an independent translation-accuracy score.
All stories retain the three chapter headings and required literal items, but only 8K, 16K
and 64K satisfy the 900–1200 Chinese-character bound. The 85K story contains 1228 Chinese
characters in both routes. Native repetitions reproduce output IDs and speculative counters
within each backend. Only 31 of 36 task/length pairs match complete IDs: native story/translation,
8K translation/logic and 16K logic differ. All 85K pairs and every JSONL pair match. No universal
token parity, artifact-quality score or quality-losslessness claim follows from these checks.

MTP reserves 8.66 GiB weights per card, DFlash 10.36 GiB. At this capacity the reported per-card
KV/workspace reservations are 1.64/0.74 GiB for MTP and 1.625/0.727 GiB for DFlash; these
figures are GPU arenas, not process RSS or aggregate two-card memory.

Three unexpected host resets interrupted this campaign. Completed results survived and only
missing repetitions were resumed. The final continuation uses 600-second active windows with
a 120-second next-request margin and **180-second idle periods**, at completed-request boundaries
and between backends. Observed DFlash active blocks before rest were 505.27 / 487.91 / 553.25 s.
Rest does not enter request times. The third reset still occurred after the three-minute-rest
policy, during the last logic measurement's prefill; cooldown is not a demonstrated reset fix.
The last saved samples showed 54/59°C and zero corrected/uncorrected GPU ECC, with no recorded
collector error or kernel Xid explaining the reset. Root cause remains undetermined; completed
measurements do not establish sustained hardware stability.

Reports and corpus remain local under `profiles/bench/quasar-qpn-projections/{all-mtp3,
all-dflash7,matrix-summary}.json` and `profiles/bench/other_inputs_20261001/cases.json`.
The read-only health logs preserve 200 ms GPU samples, per-second host/process/thread and Engine
progress, request/phase/cooldown timestamps, PCIe AER/EDAC and kernel events with synced writes.
No power limits, clocks or system settings were changed. Reproduction uses the public
[`ninfer_v100_task_bench`](../bench/README.md#stop-aware-v100-task-probes); raw outputs, telemetry,
model files and the local filming demo are not committed.
