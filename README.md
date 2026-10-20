# NInfer V100X2

[中文](README.zh-CN.md) · [Detailed measurements](docs/performance.md) · [HTTP API](docs/serving.md)

Fast Qwen3.8-27B inference on **2 × Tesla V100-SXM2 16 GB**, CUDA 12.8 / SM70.
The local cards use PCIe P2P through SXM2-to-PCIe adapters, **without NVLink**.
The measurements below are single-request; the server also runs two concurrent agent
sessions with shared prefixes, KV parking and restart survival
(see [What's new](#whats-new-2026-10-15-to-2026-10-20)).
**QUASAR NVFP4 v3 is the recommended text model**, with TP2, INT8 group-64 KV and CUDA Graphs.
It loads zero Vision weights by default; Vision is opt-in with `--vision`.

Maintainer's personal experience: “It feels in the same tier as FP8.”
The authors report GPQA-D 90.91% versus BF16 91.41%, and AIME'26 100% for both.
These are external source-checkpoint results, not our V100 conversion scores or proof of
losslessness. [Sources, independent comparison and limitations](docs/model-quality.md).

## What's new (2026-10-15 to 2026-10-20)

**Prefix reuse now survives real agent traffic.** Exact-prefix reuse no longer depends on
the client resending a byte-identical tail:

- A cancelled request retains its KV/GDN/hidden state, so ESC-then-retry resumes by append
  instead of re-prefilling.
- `--boundary-checkpoints` (with `--boundary-history N`, default 1, max 16) captures the last
  N closed message boundaries, covering clients that rewrite or drop their final message.
- Response-replay checkpoints keep the last three older checkpoints per lane, and each one
  is validated against its own token lineage as well as the lane's KV ledger, so sibling
  branches never pair one branch's state with another's pages.
- `--shared-prefix` pins the tool-definition block and the full leading instruction block
  across sessions. A system prompt that changes a per-request line still reuses the large
  tool schema; the planner takes the longest match among tools, leading block, message
  boundaries, response checkpoints and a sparse 16,384-token long-range ladder.
- The boundary a request resumed from is pinned outside the depth-limited histories, so
  alternating branches of one conversation return to their fork instead of re-prefilling
  past it (a measured case went from ~130k to ~3.5k re-prefilled tokens).

**KV parking, disk spill and restart survival.**

- `--kv-park-bytes N` copies a replaced conversation's packed KV pages and per-rank state to
  a bounded LRU host store. A returning conversation is matched by token-prefix identity and
  restores in ~0.7 s instead of re-prefilling in ~11 s (27B TP2 measurement); the restored
  lane is indistinguishable from a retained one.
- `--kv-park-spill-dir DIR` and `--kv-park-spill-bytes N` add a disk tier under the host
  budget. Startup indexes metadata sidecars without reading payloads, so a graceful restart
  (SIGINT/SIGTERM parks retained lanes first) keeps conversations instead of dropping them.
- `--shared-prefix-save FILE` / `--shared-prefix-load FILE` persist the shared prefix and
  tool anchor; `--shared-prefix-dir DIR` keeps one image per distinct prefix. Images are
  written atomically and validated (geometry, page capacity, trailing data), and older
  image/metadata versions still load.

**Two concurrent sessions with cross-lane page sharing.**

- `--max-concurrency 2` runs two lanes side by side. Physical KV pages are reference-counted:
  a request that matches another lane's pinned boundary borrows those pages, and only the
  page at the divergence is copied on write. Admission charges a shared prefix once, and a
  share plan protects its source lane from eviction.
- With sharing, the per-lane ceiling for two lanes rises from `pool/2` to `(pool + fork)/2`
  — about 228k instead of 131k tokens on the 27B TP2 pool. RAM-KV and DFlash remain excluded
  because neither has a shareable page table.
- `GET /v1/models` now returns the Anthropic model object (`max_input_tokens`, `max_tokens`,
  `created_at`, `display_name`) alongside the OpenAI fields, so clients such as
  claude-code-best size their context window from the server instead of a built-in default.
  `max_input_tokens` is the per-request sequence ceiling, not the whole KV pool.

**Vision on two 16 GB cards.**

- TP2 Vision now runs for the QUASAR artifact and alongside `--kv-park-bytes`.
- `--vision-device cpu` executes the 27-layer encoder on host threads with AVX2/FMA workers
  and sends only merged embeddings to the device.
- `--vision-kv-swap` keeps the encoder on the GPU without a startup scratch reservation:
  each image allocates its exact scratch on demand, temporarily swapping the shortfall of
  the persistent KV/runtime arena to pinned host memory at stable virtual addresses, so CUDA
  Graph replay is unaffected.
- `--vision-max-tokens` is now an ordinary option with no compiled ceiling. `unlimited`
  (also `none` or `0`) sizes the encoder by the context and media-live limits and allocates
  the request transient on demand; if that allocation fails, only that request fails with
  503 instead of stopping the engine.

**Serving ergonomics.**

- `--stream-keepalive-ms N` (default 15000, `0` disables) emits an SSE `ping` frame on
  Anthropic Messages streams while a request produces no output, so a long prefill is not
  mistaken for an idle connection by clients, proxies, tunnels or NAT.
- `--pending-timeout-ms 0` disables the admission deadline, so a request queued behind a long
  prefill waits instead of failing with `request_queue_timeout` after 30 s.
- Throughput logs now include prefill progress (`prefill_processed/prefill_prompt` tokens).

## QUASAR measurements

**85,000 actual input tokens**, 180,000 context capacity, saved repository-code task,
2560-token prefill chunks, greedy optimized-head MTP4, complete INT8 KV and CUDA Graphs.
Two cold measured requests per profile on 2026-10-13; no prefix reuse or extra request warmup.
Each publishes one prefill token plus **512 timed committed decode tokens**.
Mean ± sample standard deviation; loading and graph priming are excluded.

| Profile | Prefill tok/s | Committed wall decode tok/s | Draft acceptance |
|---|---:|---:|---:|
| Default attention / MTP4 | 1313.37 ± 3.89 | **126.89 ± 0.03** | 79.02% |
| Experimental long attention / MTP4 | 1307.41 ± 2.96 | **143.46 ± 0.01** | 76.74% |

`--volta-long-attention` keeps every historical KV entry but changes floating-point reductions
and can change output. It is **off by default**, for one SM70 TP2 Text/None/MTP request with
INT8 KV; it excludes DFlash, RAM-KV and Vision. Total request times are 68.756 / 68.585 s,
so the decode gain is not a similar whole-request gain. The fixed output window is a throughput
test, not a scored complete code solution. A local **104-question, 85K-padded GPQA** comparison
scored 88/104 default and 90/104 experimental; this partial nonstandard evaluation does not
prove quality equivalence. [Phases, numerical checks and reproduction](docs/performance.md#quasar-v3-final-long-context-decode-evaluation).

DFlash7 is useful for highly predictable structured output. Full-history INT8 KV, 98,304
capacity, chunk 1024, greedy full proposal head, no prefix reuse; one full warmup and two
measured requests. These are the original **32-record arithmetic JSONL** task, not the local
demo's JSON→JSONL conversion:

| Actual input tokens | Prefill tok/s | Committed wall decode tok/s | Draft acceptance |
|---:|---:|---:|---:|
| 118 | 580.95 ± 0.01 | **255.31 ± 0.11** | 99.05% |
| 4096 | 1577.84 ± 1.11 | 235.97 ± 0.04 | 99.05% |
| 85000 | 1184.19 ± 1.07 | 153.12 ± 0.02 | 98.30% |

The 118-token row was retested on 2026-10-13; the longer rows are the 2026-10-03 snapshot.
All six answers stop naturally at 1190 published tokens and pass exact record/arithmetic/order
checks. DFlash can be slower on prose and other tasks.
[Full QUASAR task matrix](docs/performance.md#quasar-v3-stop-aware-task-matrix).
Short prefill interval peaks near 2500 tok/s are not the phase average for an 85K request.

## llama.cpp / LM Studio comparison

There is **no current matched QUASAR-versus-llama.cpp benchmark**. The historical control used
Q4_K_M on both engines, 85,000 input tokens, requested 180,000 capacity, Q8/INT8 KV, greedy
sampling and maximum-three/minimum-zero drafts, before PCIe P2P was enabled:

| Historical Q4_K_M engine | Decode tok/s | Prefill |
|---|---:|---:|
| NInfer, staging route, two runs | 50.68 ± 0.04 | 1251.44 tok/s |
| LM Studio llama.cpp CUDA 2.33.0, one run | 35.4977 | 185.374 s |

This is historical context, **not a QUASAR speed ratio**: artifacts, output trajectories,
transport and repetition counts differ from the current results.
The NInfer windows have 128 timed tokens; the LM diagnostic has 512.
The earlier 512-input-token capacity sweep measured about 60 tok/s NInfer versus 62.5–62.8
tok/s LM Studio. [Full tables and comparison conditions](docs/llama-comparison.md).

## Build and launch

Linux, CUDA 12.8, NVIDIA driver, CMake 3.28+, C++20 compiler, Ninja, pkg-config,
FFmpeg development libraries and libcurl are required. The dependency script builds a local
prefix when system versions are too old.

```bash
bash tools/v100/build_dependencies.sh
PKG_CONFIG_PATH="$PWD/build/_deps/install/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" \
cmake -S . -B build-v100 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DBUILD_TESTING=OFF
cmake --build build-v100 --target ninfer ninfer-serve -j2
```

Download the recommended artifact without local conversion:

```bash
hf download MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer \
  qwen3_8_27b_nvfp4.ninfer SHA256SUMS \
  --revision v3 --local-dir /Models/ninfer-V100X2/quasar-v3
(cd /Models/ninfer-V100X2/quasar-v3 && sha256sum --check SHA256SUMS)
```

Start the terminal launcher (Python 3.11, standard library only):

```bash
bash tools/v100/ninfer-tui.sh
```

Select a profile, edit paths/devices/context, inspect the command, and start the API.
It shows live logs, health and Engine throughput; `s` stops its child server and `q` exits.
The default is QUASAR MTP3 / 180K, with CUDA Graphs and prefix reuse enabled.
The 143 tok/s profile is an explicit experimental MTP4 preset.
[Launcher controls and options](docs/tui.md).

Or start the API directly:

```bash
env LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  build-v100/apps/ninfer-serve /Models/ninfer-V100X2/quasar-v3/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 8080 --tp 2 --devices 0,1 --max-concurrency 1 \
  --max-context 180000 --kv-dtype int8 --prefill-chunk 2560 \
  --spec mtp --draft-tokens 3 --lm-head-draft --default-max-tokens 65536 \
  --request-log-jsonl profiles/serve.jsonl --log-stats-interval-ms 1000 --no-thinking
```

For DFlash7 use capacity 98304, chunk 1024 and `--spec dflash --draft-tokens 7`, omitting
`--lm-head-draft`. Output is bounded by remaining context.

For agent serving with two concurrent sessions, replace the concurrency/context flags and add
reuse and parking:

```bash
  --max-concurrency 2 --max-context 225280 --kv-capacity 262144 \
  --boundary-checkpoints --boundary-history 8 --shared-prefix \
  --kv-park-bytes 34359738368 --kv-park-spill-dir /models/ninfer-kv-cache
```

Optionally add `--vision --vision-kv-swap --vision-max-tokens unlimited` and
`--stream-keepalive-ms 15000`. See [What's new](#whats-new-2026-10-15-to-2026-10-20) and
[serving](docs/serving.md) for what each option does.
For CLI use `build-v100/apps/ninfer` with the same inference flags and `--prompt TEXT --max-new N`.
Installed GPU count does not fix runtime width: `--tp 2 --devices 4,6` selects a pair on an
eight-card host. With `CUDA_VISIBLE_DEVICES=4,6`, use logical devices `0,1`.
[Test device selection](tests/README.md#build-and-run).

## Other configurations

| Document | Scope |
|---|---|
| [TP4 / NVLink](docs/tp4.md) | separate four-card host, official NVFP4/native FP8, API and 1Cat controls |
| [Other V100 measurements](docs/performance-v100.md) | official NVFP4, Q4_K_M, P2P/staging controls and prefix tests |
| [RAM-KV / PP2 experiments](docs/experiments.md) | approximate retrieval and startup-fixed layer pipeline |
| [Alternative models / Vision](docs/alternatives.md) | Q4_K_M conversion and experimental official-NVFP4 Vision |
| [V100 implementation](docs/v100-implementation.md) | kernel, transport and numerical verification details |
| [Inherited upstream results](docs/performance-upstream.md) | other hardware/models, concurrency and extended-context campaigns |
| [Documentation index](docs/README.md) | CLI, API, GUI, tests and maintainer references |

The recommended profile uses **tensor parallelism (TP2)**. PP2 is a separate startup-fixed
experiment; the published TP2 numbers do not include pipeline-parallel gains.

## Acknowledgements and license

V100X2 implementation work builds on [Neroued/ninfer](https://github.com/Neroued/ninfer), the
RTX 3060 TP2 fork, and [geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100).
Volta NVFP4 and TP2 behavior was cross-checked against
[plus1998/Ninfer-V100-Duo](https://github.com/plus1998/Ninfer-V100-Duo). The prefill investigation
consulted [1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM). DFlash2 follows
[Inco AI's implementation](https://inco.ai/blog/dflash2/) and its
[Qwen3.8-27B draft checkpoint](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2).
[KVMem](https://github.com/kvmem/kvmem-llama.cpp) informed the optional RAM-KV experiment.
The native implementation adapts Q/K mean-vector retrieval to chronological paged KV; selective-history
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
