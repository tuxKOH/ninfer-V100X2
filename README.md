# NInfer V100X2

This fork targets Qwen3.8-27B Q4_K_M text inference on two Tesla V100-SXM2 16 GB cards with
CUDA 12.8 (`sm_70`). The default launcher uses tensor parallelism across both cards, a
180,000-token context capacity, INT8 group-64 KV cache, CUDA Graphs, and MTP with up to three
draft tokens and the optimized proposal head. The capacity is an allocation limit; a performance
result must also state how many prompt tokens were actually filled.

NInfer is a from-scratch C++/CUDA engine descended from
[Neroued/ninfer](https://github.com/Neroued/ninfer). This checkout combines the dual-GPU path from
the RTX 3060 fork with Volta work from
[geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100). Ampere (`sm_86`) and early Ada
(`sm_89`) builds remain available. The inherited RTX 5090 results and YaRN 1M-context results
below describe their original hardware and artifacts; they do not establish V100X2 performance.
See [NOTICE](NOTICE) and the
[TP2 architecture reference](docs/maintainer/tp2-yarn-1m.md) for upstream attribution and design.

At 85K occupied prompt tokens and 180K capacity, the fixed code-generation workload measured
**53.41 ± 0.064 committed decode tok/s** across three 512-token decode windows, **18.68% above**
the user's approximately 45 tok/s baseline. A separate **512-token prompt** code workload at the
same 180K capacity measured **60.03 ± 0.057 tok/s** across three 256-token decode windows.
Values are means ± sample standard deviations; all six output windows were EOS/EOG-free.

Both profiles use INT8 KV, optimized MTP3, CUDA Graphs and verified CUDA host-staged TP2. Original
weight codes/scales are preserved, with independent operator oracles and real-model MTP regression
checks. These results apply to the measured code workloads. The short-input result exceeds the
reported 57 tok/s peak, whose unspecified context occupancy prevents a matched comparison;
the 85K result remains below 57 tok/s. See
[V100X2 measurement](docs/performance.md#v100x2-measurement-and-acceptance).

NInfer deliberately supports a closed set of model artifacts instead of acting as a general model
runtime:

| Model | Weights | NInfer artifact | Size | SHA-256 |
|---|---|---|---:|---|
| [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) | `groupwise-int` | `qwen3_6_27b.ninfer` | 17,495,365,888 bytes (16.29 GiB) | `7b51600ffd10632b9660f56085efdd9b751d79733ad32036a652234b64bebe7b` |
| [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | 18,324,064,000 bytes (17.07 GiB) | `bce5f00d066c0f20f1317bf1fdcb458264cf95837c3b1f3fbec163694627893a` |
| [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) | `groupwise-int` | `qwen3_8_27b.ninfer` | 18,210,531,328 bytes (16.96 GiB) | `eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e` |
| [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | 21,492,695,040 bytes (20.02 GiB) | `bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32` |
| Qwen3.8-27B GGUF Q4_K_M (local V100 profile) | `gguf-q4-k-m` | `qwen3_8_27b_q4_k_m.ninfer` | local conversion | local artifact |
| [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | 22,783,246,080 bytes (21.22 GiB) | `1fb9ea0b5b8561e49d9604115ec89e5d9f2b6f6434e32c37c57fffd480a325d2` |

Qwen3.6-27B exposes two registered weight profiles, and Qwen3.8-27B adds the local GGUF-derived
profile to its two published profiles. The version-2 artifact
identity selects the profile without a separate runtime flag; Qwen3.8 uses target key
`qwen3_8_27b` while sharing the 27B execution package. The Qwen3.6 `nvfp4` profile uses W4A4 Tensor
Core MMA for prefill and A16 NVFP4 kernels for decode. The Qwen3.8 `nvfp4` profile preserves its
source's mixed allocation: NVFP4 MLP weights in Text layers 0–55 and row-scaled FP8 for the token
embedding, attention input/output projections, GDN Q/K/V/Z and output projections, output head, and
remaining MLP weights. All four 27B artifacts retain the same Text, Vision, MTP, prefix-reuse, CLI,
and serving routes.

The local `gguf-q4-k-m` profile comes from the Qwen3.8 Q4_K_M GGUF in LM Studio's model directory.
Its original Q4_K/Q6_K codes and embedded scales are preserved without requantization. Control
tensors and frontend resources are converted to NInfer's representation; those transformations
require numerical and behavioral checks before making an end-to-end quality claim. See the
[GGUF artifact contract](docs/maintainer/qwen3.8-27b-artifact.md#14-preserved-gguf-q4_k_m-artifact).
It supports Text and MTP through the same Engine route; its embedded GGUF Vision objects are
validation-only and `--vision` is rejected for this identity. The artifact is intentionally kept
outside the repository because it is an 18 GB generated model file.

## Inherited RTX 5090 performance

The following published measurements cover the three Qwen3.6 artifact profiles and the Qwen3.8-27B NVFP4
profile. The Qwen3.8-27B `groupwise-int` profile is supported by current NInfer builds but is not
yet included in a published benchmark campaign.

### Concurrent MTP3 decode

Saturated decode was measured on an RTX 5090 with INT8 group-64 KV cache, CUDA Graphs, MTP3, and
one 8,192-token generation per active request. The values below are aggregate committed decode
throughput from complete one-second intervals in which the actual decode batch remained equal to
the configured concurrency. MTP acceptance is aggregated over the complete request wave. Each
concurrency cell reports `decode tok/s / MTP acceptance`; profiles should be read independently.

| Model profile | C=1 tok/s / accept | C=2 tok/s / accept | C=4 tok/s / accept | C=8 tok/s / accept | C8 / C1 |
|---|---:|---:|---:|---:|---:|
| Qwen3.6-27B `groupwise-int` | 185.8 / 68.2% | 247.0 / 69.0% | 309.5 / 68.4% | 535.0 / 68.3% | 2.88× |
| Qwen3.6-27B `nvfp4` | 202.4 / 69.3% | 399.7 / 71.4% | 699.7 / 69.3% | 1,146.9 / 68.6% | 5.67× |
| Qwen3.6-35B-A3B `groupwise-int` | 593.0 / 67.2% | 877.7 / 68.2% | 1,166.0 / 69.8% | 1,313.8 / 67.3% | 2.22× |
| Qwen3.8-27B `nvfp4` | 143.8 / 48.9% | 267.6 / 48.1% | 461.1 / 45.8% | 766.6 / 46.0% | 5.33× |

At C=8, Qwen3.6-35B-A3B reaches **1,313.8 aggregate decode tok/s**. Qwen3.6-27B NVFP4 reaches
**1,146.9 tok/s** and **5.67×** its C=1 throughput. Qwen3.8-27B NVFP4 has **45.8–48.9%** MTP
acceptance, versus **67.2–71.4%** across the other measured profiles, so aggregate committed
throughput reflects both execution performance and speculative acceptance.

### Single-request serving

The single-request corpus was measured on the same GPU with INT8 group-64 KV cache, CUDA Graphs,
and a 1,024-token prefill chunk. Each reported fixture uses five fixed seeds after server warm-up.
Targets and weight profiles are reported independently rather than as cross-target comparisons.
Requests were submitted serially to a persistent server. The Qwen3.8-27B NVFP4 MTP0 results use the
same dedicated serial corpus runner as the Qwen3.6 profiles; its MTP3 results come from the C=1 point
of the fixed concurrent-corpus campaign documented in [Performance](docs/performance.md).

**Qwen3.6-35B-A3B**

- MTP0 at a 7,680-token prompt: **15,544.3 prefill tok/s** and **271.1 decode tok/s**.
- MTP0 at a 260,096-token prompt: **5,157.1 prefill tok/s** and **188.2 decode tok/s**.
- MTP3 long reasoning: **620.3–726.2 decode tok/s** with **72.7–82.8% acceptance**.
- MTP3 structured output: **770.9 decode tok/s**, **89.1% acceptance**, and **3.67 tokens/round**.

**Qwen3.6-27B (`groupwise-int`)**

- MTP0 at a 7,680-token prompt: **3,218.1 prefill tok/s** and **77.6 decode tok/s**.
- MTP0 at a 260,096-token prompt: **1,614.8 prefill tok/s** and **54.8 decode tok/s**.
- MTP3 long reasoning: **161.9–175.4 decode tok/s** with **73.4–78.8% acceptance**.
- MTP3 structured output: **193.0 decode tok/s**, **88.7% acceptance**, and **3.66 tokens/round**.

**Qwen3.6-27B (`nvfp4`)**

- MTP0 at a 7,680-token prompt: **11,191.5 prefill tok/s** and **86.4 decode tok/s**.
- MTP0 at a 260,096-token prompt: **2,510.6 prefill tok/s** and **59.9 decode tok/s**.
- MTP3 long reasoning: **213.1–231.0 decode tok/s** with **76.3–81.1% acceptance**.
- MTP3 structured output: **252.2 decode tok/s**, **89.8% acceptance**, and **3.69 tokens/round**.
- Against groupwise-int on the same corpus and runtime options: **3.48× the 7,680-token prefill
  throughput**, **1.55× the 260,096-token prefill throughput**, and **30–32% higher MTP3 decode
  throughput**.

**Qwen3.8-27B (`nvfp4`)**

- MTP0 at a 7,680-token prompt: **8,340.4 prefill tok/s** and **71.2 decode tok/s**.
- MTP0 at a 260,096-token prompt: **2,203.1 prefill tok/s** and **52.9 decode tok/s**.
- MTP3 long reasoning: **151.4–195.2 decode tok/s** with **56.2–76.0% acceptance**.
- MTP3 structured output: **219.8 decode tok/s**, **90.8% acceptance**, and **3.72 tokens/round**.

See [Performance](docs/performance.md) for the full methodology, variability, reproduction command,
and per-fixture results.

## Evaluation

Capability scores were measured through NInfer's OpenAI-compatible serving route with thinking
enabled, MTP=3, and EvalScope 1.9.0 (0-shot, rule scoring, one sample per problem):

| Model profile | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

The Qwen3.6 rows used temperature 0.6 and presence penalty 1.0; the Qwen3.8-27B rows used
temperature 1.0 and presence penalty 0.0. The multimodal columns (ERQA and RealWorldQA) ran with
`--vision` at a 81,920-token context limit; the text columns used a 262,144-token limit except
Qwen3.8-27B NVFP4, which needs 252,928 to fit the RTX 5090 after weights.

These are single-sample results under that NInfer evaluation profile, not pass@k. See the model
cards and [full performance document](docs/performance.md) for correct/total counts and evaluation
notes.

## Requirements

NInfer currently requires:

- 64-bit Linux;
- two Tesla V100-SXM2 16 GB cards (`sm_70`) for the V100X2 profile; the retained compatibility
  builds target Ampere `sm_86` and early Ada `sm_89`;
- NVIDIA driver support for the selected GPU and CUDA 12.8 for Volta; CUDA 13 no longer compiles
  Volta;
- CMake 3.28 or newer and a C++20-capable host compiler;
- `pkg-config`;
- FFmpeg development libraries: `libavformat >= 60`, `libavcodec >= 60`,
  `libavutil >= 58`, and `libswscale >= 7`;
- `libcurl >= 7.85`;
- Ninja, when using the commands below.

The build accepts `70` (default), `86`, or `89` in this checkout. RTX 5090 `120a` remains an upstream
configuration and is not accepted by this checkout's CMake. There is no install target
or packaged binary distribution; NInfer is run from its source build tree.

## Build

Clone the V100X2 fork and select the CUDA 12.8 compiler explicitly:

```bash
git clone https://github.com/tuxKOH/ninfer-V100X2.git
cd ninfer-V100X2
```

If the host lacks the required FFmpeg or curl development versions, build the private dependencies
first. The helper installs under `build/_deps/install`:

```bash
tools/v100/build_dependencies.sh
```

Configure with that prefix (an absent prefix is harmless when system dependencies already suffice):

```bash
PKG_CONFIG_PATH="$PWD/build/_deps/install/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" \
cmake -S . -B build-v100 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build-v100 -j
```

The V100X2 launcher selects the local Qwen3.8-27B Q4_K_M artifact and the requested long-context
profile (`--tp 2`, `--max-context 180000`, INT8 group-64 KV, CUDA Graphs, MTP draft window 3,
optimized proposal head via `--lm-head-draft`) by default. Accepted draft counts can range from
zero through three in each round. The target verifier still uses the full vocabulary. Pass the
normal CLI options, including `--prompt` or `--messages`, after the launcher:

```bash
tools/v100/ninfer-v100x2.sh \
  --prompt "Explain prefill and decode in three sentences." \
  --max-new 128 --greedy --no-thinking
```

Override the artifact, device pair, context capacity, or draft window with
`NINFER_V100X2_ARTIFACT`, `NINFER_V100X2_DEVICES`, `NINFER_V100X2_MAX_CONTEXT`, and
`NINFER_V100X2_DRAFT_TOKENS`. Set `NINFER_V100X2_PROPOSAL_HEAD=full` to use the full proposal
head; the only accepted values are `full` and `optimized` (the default). The launcher does not
impose a low host-CPU affinity; the executor blocks when no request is ready, while an external
cgroup/CPU quota can still cap the process if a host requires it.

The V100 configuration builds:

```text
build-v100/apps/ninfer
build-v100/apps/ninfer-serve
```

Tests, benchmarks, and maintainer tools are excluded from the default build.

The inherited `Dockerfile` uses CUDA 13.1 and has not been retargeted for V100. Use the CUDA 12.8
source build above for this profile.

## Convert the local LM Studio model

Conversion takes the exact Q4_K_M GGUF and its BF16 companion vision file; it writes a native
`.ninfer` artifact and a conversion report. It does not download a replacement checkpoint. Adjust
the explicit paths to your installation:

```bash
python3 -m tools.convert.qwen3_8_27b.convert_gguf \
  --model /home/z/.lmstudio/models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf \
  --mmproj /home/z/.lmstudio/models/lmstudio-community/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-BF16.gguf \
  --out /Models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer
```

Use a Python environment containing NumPy. The runtime accepts the resulting `.ninfer`, not the
source GGUF. When this artifact already exists, run the launcher directly.

## Download a model

Use the Hugging Face CLI to download one of the registered artifacts:

```bash
hf download neroued/Qwen3.6-27B-NInfer \
  qwen3_6_27b.ninfer \
  --local-dir models

# Or the 27B NVFP4 weight variant:
hf download neroued/Qwen3.6-27B-nvfp4-NInfer \
  qwen3_6_27b_nvfp4.ninfer \
  --local-dir models

# Or Qwen3.8-27B:
hf download neroued/Qwen3.8-27B-NInfer \
  qwen3_8_27b.ninfer \
  --local-dir models

# Or Qwen3.8-27B NVFP4:
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer \
  --local-dir models

# Or:
hf download neroued/Qwen3.6-35B-A3B-NInfer \
  qwen3_6_35b_a3b.ninfer \
  --local-dir models
```

Current NInfer builds accept only the version-2 artifact container, and all five downloads above
are version 2. Migration applies only to Qwen3.6 artifacts downloaded before their version-2
publication; both Qwen3.8-27B profiles were published directly as version 2. Migrate an older exact
local file in place:

```bash
python3 -m tools.artifact.migrate_v1_to_v2 models/qwen3_6_27b.ninfer
```

Use the same command with `qwen3_6_27b_nvfp4.ninfer` or `qwen3_6_35b_a3b.ninfer` for those
artifacts. The migration updates only container metadata; it does not rewrite the weight payload.
Alternatively, download the current version-2 file again from its Hugging Face repository.

Each `.ninfer` file contains the weights and frontend resources needed by NInfer. It is not a
Transformers checkpoint, Safetensors distribution, or GGUF file.

Each artifact is complete, while GPU residency is fixed at process startup. Speculative decoding is
disabled by default, so MTP/DFlash state and the optimized proposal head are not uploaded.
Vision is also disabled by default, so its weights, Vision scratch phase, and frozen
request-transient allocation are omitted. Add `--vision` to the CLI or server process that must
accept image or video input. Disabled capabilities cannot be enabled by a later request. DFlash is
available only for the 35B-A3B target and is text-only.

## Run the CLI

For V100X2 use `tools/v100/ninfer-v100x2.sh` as shown above. The direct CLI and server examples
below describe the inherited published artifacts; substitute the V100 artifact, `build-v100`
binary, and explicit TP2/INT8/MTP3 options when running this profile.

```bash
./build/apps/ninfer models/qwen3_6_27b.ninfer \
  --prompt "Explain prefill and decode in three sentences." \
  --max-context 16384 \
  --max-new 256 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

Use `--messages FILE` instead of `--prompt` for chat history, images, or videos:

```bash
./build/apps/ninfer models/qwen3_6_27b.ninfer \
  --messages examples/cli/messages/image_chart.json \
  --max-context 8192 \
  --max-new 128 \
  --vision
```

Answer content is written to stdout. Loading progress, reasoning, timing, throughput, memory, and
speculative-decoding statistics are written to stderr. See the [CLI guide](docs/cli.md) and
[committed examples](examples/cli/) for structured input and runtime options.

## Run the HTTP server

```bash
./build/apps/ninfer-serve models/qwen3_6_27b.ninfer \
  --max-context 16384 \
  --kv-capacity auto \
  --max-concurrency 2 \
  --spec mtp --draft-tokens 3 \
  --lm-head-draft
```

The public model ID defaults to the artifact's `identity.model_id`; use `--model-id` only to
publish a deployment-specific alias.

Then send an OpenAI-style request:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-27b",
    "messages": [{"role": "user", "content": "Reply with one short sentence."}],
    "max_tokens": 64
  }'
```

The server also implements OpenAI Responses Core (typed Items, semantic SSE, local continuation
state, and function calls) plus Anthropic Messages, token counting, and multimodal input. See
[HTTP serving](docs/serving.md).

## Dual-GPU (TP2) and YaRN 1M context

This section retains the original RTX 5090 campaign and commands. It is not a V100 capacity or
performance claim; V100X2 uses the 180,000-token profile described above.

`--tp 2` splits one resident model across two RTX 5090s, and `--rope yarn` raises the addressable
context ceiling from the registered native 262,144 tokens to 1,048,576. The two features are
independent -- TP2 halves per-card weight and KV residency at any context, YaRN extends positions
at either `--tp` width -- but 1,048,576 tokens only fits when both are used together with INT8 KV.

TP2 runs one process, one resident model and two CUDA devices without distributed serving.
The RTX 5090 pair in this campaign has no NVLink. TP2 is implemented for the 27B execution package
(`qwen3.6-27b` and `qwen3.8-27b`); `qwen3.6-35b-a3b` has no tensor-parallel
path and rejects `--tp 2` at startup. Every measurement below was taken on the Qwen3.8-27B NVFP4
artifact.

### Usage

```bash
# 1,048,576-token context, both GPUs, INT8 KV, MTP3 speculative decoding
./build/apps/ninfer-serve models/qwen3_8_27b_nvfp4.ninfer \
  --tp 2 --devices 0,1 \
  --rope yarn --yarn-factor 4.0 --yarn-origin 262144 \
  --max-context 1048576 --kv-dtype int8 --kv-capacity auto \
  --max-concurrency 1 \
  --spec mtp --draft-tokens 3 --lm-head-draft
```

The same flags drive the CLI; add `--no-thinking` when a short `--max-new` budget must reach the
answer channel, because at this checkpoint's default thinking mode a 16-token budget is consumed
entirely inside the reasoning stream:

```bash
./build/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --tp 2 --devices 0,1 \
  --rope yarn --yarn-factor 4.0 --yarn-origin 262144 \
  --max-context 1048576 --kv-dtype int8 --kv-capacity auto \
  --messages long_prompt.json --max-new 256 --no-thinking
```

- `--tp 2` requires an explicit `--devices A,B` naming two distinct devices of the same compute
  capability. `--tp 1` remains the default and is unchanged: `scripts/tp1-regression.sh` compares
  this build's greedy output against token streams recorded from upstream `ninfer` at base commit
  `feaf4dd`, over a short chat, a 2,191-token instruction and a 28,677-token document, and requires
  the token ids, the generated text and the deterministic summary rows to be byte-equal. Its scope
  is exactly that: greedy text decode, `--tp 1`, `--rope native`, NVFP4 weights, `qwen3.8-27b`, MTP
  off, concurrency 1. See `tests/data/tp1-golden/MANIFEST.md`.
- `--rope yarn` needs `--yarn-origin` to equal the artifact's registered native capacity
  (`262144`); `--yarn-factor` is a finite value in `[1.0, 64.0]` and `origin x factor` must be a
  whole token count at or below `1048576`.
- `--kv-dtype int8` is mandatory at 1M: BF16 KV needs four times the pool and does not fit.
- `--max-concurrency 1` is arithmetic, not policy, at 1M -- one sequence costs 16.66 GiB per device
  without MTP and 17.69 GiB with it, so a second slot cannot fit on a 32 GiB card.
- MTP speculative decoding (`--spec mtp --draft-tokens 1..5`, optionally `--lm-head-draft`) works
  at `--tp 2` including at 1M. `--spec dflash` and `--vision` are rejected at `--tp 2`.

See the [CLI guide](docs/cli.md) and [HTTP serving](docs/serving.md) for the full option contract.

### Memory, per GPU

Qwen3.8-27B NVFP4, `--tp 2 --devices 0,1 --kv-dtype int8 --kv-capacity auto`, one active request.
`Resident` is `nvidia-smi` per-process memory, which was flat across every sample of every run --
a 1M-token prefill does not push residency above the value chosen at load.

| Context | MTP | Weights | Sequence (KV + GDN state) | Workspace | Reserved (load summary) | Resident (`nvidia-smi`) |
|---:|---|---:|---:|---:|---:|---:|
| 262,144 | off | 10.08 GiB | 4.28 GiB | 0.18 GiB | — | **15.04 GiB** |
| 262,144 | MTP3 | 10.46 GiB | 4.54 GiB | 0.19 GiB | — | **15.69 GiB** |
| 1,048,576 | off | 10.08 GiB | 16.66 GiB | 182.81 MiB | 26.93 GiB | **27.41 GiB** |
| 1,048,576 | MTP3 | 10.46 GiB | 17.69 GiB | 192.93 MiB | 28.42 GiB | **28.84 GiB** |

The KV pool is 16.5 KiB per token per device at INT8 group-64, quantization-scale planes included,
which is the 16.50 GiB pool at 1,048,576 tokens.
Turning MTP3 on costs a **measured 1.49 GiB of reserved memory per device at 1M and 0.65 GiB at
262k** -- it is not a constant. Its two dominant terms are **0.38 GiB of head weights**, fixed at
any window, and **1.03 GiB of MTP KV per 1M tokens of window**; the remainder of each measured
delta is workspace and sequence-arena rounding. The KV term is one full extra attention layer's
worth of KV over the window -- one sixteenth of the text pool -- which is what a one-layer MTP head
costs. The 262,144-token rows were measured through `ninfer-serve`'s startup record and
`nvidia-smi`, which is why they carry no CLI load-summary `reserved` row.

At 1M with MTP3 the headroom to a 30 GiB per-device budget is **1.16 GiB**, the tightest shipped
configuration. For comparison, the same artifact at `--tp 1` needs 27.9 GiB on one card for a
252,928-token window with MTP off and 29.16 GiB with MTP3, and cannot reach 262,144 at all.

### Measured performance

**Every figure carries a per-GPU power limit.** The campaign was measured at a **400 W** cap (the
minimum settable limit on these cards); the publishable set was then re-measured with both cards at
**575 W** (vendor defaults are 600 W and 575 W; maximum 600 W). Do not quote any figure below
without its power condition.

Single request, INT8 KV, CUDA Graphs on, greedy decoding, byte-identical prompt on both widths:

| Workload | TP1 @400 W | TP2 @400 W | TP1 @575 W | TP2 @575 W |
|---|---:|---:|---:|---:|
| 249,955-token prompt, MTP off — prefill | 2,269.8 | **2,680.1** | 2,484.2 | **2,787.0 tok/s** |
| 249,955-token prompt, MTP off — decode | 52.35 | **75.18** | 53.95 | **75.32 tok/s** (1.40x) |
| 249,955-token prompt, MTP3 — decode | 101.7 | **152.1** | 113.60 | **159.39 tok/s** |
| 249,955-token prompt, MTP3 — draft acceptance | 50.83% | **57.96%** | 50.83% | **57.96%** |
| 249,955-token prompt — time to first token | 111.0 s | **93.7 s** | 101.0 s | **90.0 s** |
| 536-token reasoning prompt, MTP3 — decode | 152.0 | **189.5** | — | — |
| 652,955-token prompt, MTP off — prefill / decode | does not fit | **1,348.4 / 58.22** | does not fit | not re-measured |
| 1,045,954-token prompt, MTP off — prefill / decode | does not fit | **928.9 / 46.39** | does not fit | **975.1 / 48.08** |
| 1,045,954-token prompt, MTP3 — decode (512 tokens) | does not fit | — | does not fit | **100.54 tok/s** at 56.41% |

Lifting the cap helps TP1 more than TP2, and the reason is visible in sampled draw: one card running
the whole model saturates its limit (peak 575.5 W), while two cards sharing it peak at 391 and
406 W. The TP2-over-TP1 decode advantage therefore narrows from **1.44x at 400 W to 1.40x at
575 W** -- TP2 is still faster, and it gets there inside a much smaller power envelope.

At 575 W the 1,045,954-token prefill takes **17.9 minutes** per request, against 18.8 at the 400 W
cap. Decode degrades smoothly with context rather than falling off a cliff: at the 400 W cap,
97.9 tok/s at 8k, 58.2 at 653k and 46.4 at 1,046k; the 1,046k point rises to 48.1 at 575 W. Only
the 250k and 1,046k tiers were re-measured at 575 W. At long context TP2 is *faster* than TP1, not
merely larger, because per-card weight and KV traffic halve while the cross-device collectives cost
about 0.2 ms per token under CUDA Graphs.

Saturated concurrent decode at a 262,144-token window, `--tp 2`, aggregate committed tok/s:

| Concurrency | MTP off @400 W | MTP3 @400 W | MTP off @575 W | MTP3 @575 W |
|---:|---:|---:|---:|---:|
| 1 | 93.5 | 172.4 | 94.1 | 177.8 |
| 2 | 183.0 | 287.0 | — | — |
| 4 | **314.3** (3.36x) | **466.0** (2.70x) | **320.3** (3.40x) | **475.2** (2.67x) |

Per-GPU residency stayed at or below 15.64 GiB at C=4.

### Compared with vLLM

On byte-identical 250k / 653k / 700k-token prompts with **both cards capped at 500 W** -- a third
power condition, not comparable with the 400 W and 575 W rows above -- vLLM 0.25.1 (TP2, FP8 KV,
YaRN through `--hf-overrides`, MTP3) prefills **1.17-1.32x faster**, while NInfer decodes **1.41x
faster at 250k and 2.5-2.8x faster at 653k and 700k**, because vLLM's MTP acceptance is exactly
**0%** past its native 262,144-token window where NInfer's stays at 51-60%. NInfer also holds a
**38% larger window in less memory**: 1,048,576 tokens at 27.41 GiB per GPU against vLLM's 759,297
at 28.90 GiB. At a 512-token answer the request is almost all prefill, so vLLM finishes first at
every tier; NInfer wins beyond roughly 4,800 / 7,600 / 8,800 output tokens. The two engines ran
different NVFP4 quantizations of different fine-tunes and different KV dtypes, and no quality claim
is made -- the full table, method and caveats are in
[Performance](docs/performance.md#cross-engine-comparison-against-vllm-nvfp4-500-w-per-gpu).

### Retrieval

Needle-in-a-haystack, five depths (10/30/50/70/90%) x two independent haystacks, rule-scored exact
match, thinking disabled, greedy:

| Engine and configuration | Haystack | Prompt tokens | Retrieved |
|---|---|---:|:-:|
| NInfer TP2, native rope | 262k, **tiled** corpus | 259,954 | **10 / 10** |
| NInfer TP2 + YaRN x4 | 653k, distinct text | 652,954-652,955 | **10 / 10** |
| NInfer TP2 + YaRN x4 | **1,046k, distinct text** | 1,045,954-1,045,955 | **10 / 10** |
| vLLM control (model ceiling) | 653k, distinct text | 652,954-652,955 | **10 / 10** |

Three qualifications travel with that table:

- **The 262k row uses a tiled haystack.** The stock corpus is 2.7 MB (644 KB of English essays plus
  a Chinese novel), so any English tier past roughly 150k tokens repeats the corpus -- at 262k each
  window occurs about twice. Retrieval stays valid because the needle is unique, but it is an
  easier task than the 653k and 1,046k rows, whose haystacks are purpose-built distinct text with
  200 of 200 sampled windows occurring exactly once.
- **The vLLM control is a different checkpoint** (`orcarouter/Qwen3.8-27B-Uncensored-FP8`, FP8
  weights, FP8 KV, speculation on) against NInfer's NVFP4 conversion of a different fine-tune. The
  two rows share a model family and a YaRN configuration, not weights. The row establishes the
  model-side retrieval ceiling under this YaRN configuration; it is not an engine-versus-engine
  comparison.
- **10/10 is not a demonstration of a high per-depth rate.** The 1M grid ran two samples per depth;
  for 10 successes in 10 trials the exact one-sided 95% Clopper-Pearson lower bound is 74%.
- **A needle just past 262k does not by itself prove YaRN works.** With the YaRN descriptors
  nulled, a 270k needle still resolved -- the model tolerates roughly 10% extrapolation past its
  trained ceiling unaided. The >262k legs of the real-weights YaRN test therefore establish
  *position addressability*, not YaRN quality. The 653k and 1,046k retrieval rows are what
  demonstrate YaRN's value, being at genuine multiples of the native window.

Retrieval, not recall: with an invented needle -- an invented place and an invented activity that
cannot be in any training set -- substituted at depth 50 of a 1,045,956-token prompt, the model
reproduced the sentence verbatim.

### Soak

All timings in this subsection were taken at the 400 W per-GPU cap (see
[Measured performance](#measured-performance) above); the soak was not re-run at 575 W.

Two consecutive passes decoded from a 949,885-token prompt to the exact 1,048,576-token ceiling:
98,692 generated tokens each, `finish reason context-capacity`, 52 minutes of wall clock per pass,
prefill 1,011.89 and 1,009.97 tok/s, decode 45.49 and 45.39 tok/s, per-device residency flat at
28,070 MiB (span 4 MiB), and the two passes' 98,692-token id streams and 380,672-byte reply texts
hash identically. No OOM, no NaN: ten logit-sanity windows sampled every 10k tokens kept top-1
share at 6.43-6.60% with no out-of-domain ids.

Greedy decoding degenerates on this prompt at this context length, by two distinct mechanisms, and
the soak stream is therefore stability and determinism evidence rather than a sample of 1M-context
generation quality. Without a speculative backend and with `--ignore-eos`, the model finishes an
answer turn, the suppressed end-of-turn token lets it answer again, and it settles into a
1,903-token turn cycle repeated 46 times byte for byte. With MTP3 the ~950k greedy stream contains
no end-of-turn token at all and still collapses, into a 187-token content-level loop whose first
repeat begins at generated index **1,201** -- that one is plain greedy degeneration, not an
`--ignore-eos` artifact. Sampled decoding at temperature 0.8 does not loop.

### YaRN reference and drift guard

NInfer's YaRN frequency correction is computed to match vLLM as deployed, including vLLM's
multimodal rotary path: correction range `(16, 24)` and an `mscale` of 1.13863 applied to `cos`/`sin`
over the 64 rotary dimensions of each 256-dimension head. The reference values are checked in at
`tests/core/data/yarn_ref_4x.json` (recorded against vLLM 0.25.1), and
`ninfer_qwen3_6_yarn_rope_drift_test` regenerates them from the installed vLLM and fails if either
the numbers or the vLLM version drift. Without a vLLM environment the test skips. To regenerate the
reference after a deliberate vLLM upgrade:

```bash
VLLM_LOGGING_LEVEL=WARNING "$NINFER_VLLM_PYTHON" tools/tp2/dump_yarn_ref.py \
  > tests/core/data/yarn_ref_4x.json
ctest --test-dir build -R ninfer_qwen3_6_yarn_rope_drift_test --output-on-failure
```

`NINFER_VLLM_PYTHON` points at the Python interpreter of the vLLM environment; the test falls back
to a known local path when the variable is unset, and `NINFER_YARN_REF_JSON` overrides the compared
file.

### Test environment variables

The dual-GPU and YaRN tests are opt-in the same way the rest of the artifact-gated suite is: each
reads an environment variable, falls back to a local default path, and skips (rather than fails)
when the resource is not present.

| Variable | Used by | Default |
|---|---|---|
| `NINFER_QWEN3_8_27B_NVFP4_WEIGHTS` | the sharded-materialization and TP2 real-Engine CTest routes | `/home/pc/models/ninfer-38/huihui-nvfp4/qwen3_8_27b_nvfp4.ninfer` |
| `NINFER_VLLM_PYTHON` | `ninfer_qwen3_6_yarn_rope_drift_test`, and `tools/tp2/dump_yarn_ref.py` above | `/home/pc/Projects/vllm/unsloth-nvfp4-env/bin/python` |
| `NINFER_YARN_REF_JSON` | `ninfer_qwen3_6_yarn_rope_drift_test`, to compare against a scratch reference instead of the committed one | `tests/core/data/yarn_ref_4x.json` |

`scripts/tp1-regression.sh` takes the artifact as its first positional argument instead, and
`TP1_GPU` selects the physical GPU it runs on.

### Limitations

- **Vision is `--tp 1` only.** The Vision encoder runs on the primary device against replicated
  weights and has no split path, so `--tp 2 --vision` is rejected at startup. YaRN is likewise
  rejected together with `--vision`, because the encoder ropes 2-D image-grid positions.
- **DFlash is unchanged and is rejected at `--tp 2`.** It remains a 35B-A3B text-only backend, and
  that target has no tensor-parallel path at all.
- **No NVLink, and no peer-to-peer on GeForce.** `cudaDeviceCanAccessPeer` reports 0 between two
  RTX 5090s, so the collectives are host-staged asynchronous copies over PCIe rather than direct
  peer copies. This is a measured property of the hardware, not a configuration choice. A decode
  token costs 128 reduces plus one logit all-gather; a 10 KiB reduce measures about 16 us, and
  under CUDA Graphs the whole collective set costs roughly 0.2 ms per token (both at the 400 W
  per-GPU cap).
- **MTP prefix reuse resets at `--tp 2`.** Resuming a prefix drives the MTP head from a retained
  target hidden state that only the primary device holds, so `--tp 2 --spec mtp` downgrades every
  reuse to a full prefill. The answer is unchanged and no request fails; the saving is lost, which
  matters for multi-turn conversation at long context.
- **MTP is output-equivalent up to near-tie argmax flips, not bit-identical.** A verify round
  evaluates the target model over `K+1` columns at once and an ordinary round over one, which
  selects different GEMM shapes; greedy MTP-on and MTP-off streams can therefore diverge on a
  near-tie token. Every committed token is still one the target model's own argmax selected. On the
  1,046k needles the MTP3 and MTP-off answers are identical token for token; on a 949,885-token
  soak stream they first differ at generated index 45. Measured like for like at 575 W -- same
  1,045,954-token prompt, same 512-token budget -- MTP3 decodes **2.18x** faster than MTP off.
- **1,048,576 tokens is a one-slot configuration.** `--max-concurrency` must be 1; concurrency at
  extended context requires coming down from the ceiling (roughly 500k for two slots at INT8 KV).
- **`--ignore-eos` is a diagnostic flag.** It exists for fixed-length soak and throughput work.
  Generation past the end-of-turn token is off-distribution and is not a product output. It is not
  the only route to a degenerate stream: greedy decoding at this context length also loops on its
  own content, with the flag off.
- **A full-ceiling prefill costs about 18 minutes** (17.9 at 575 W, 18.8 at the 400 W cap), and 1M
  decode runs at roughly 48 tok/s, or 100 tok/s with MTP3. Retrieval is not the binding constraint
  at 1M; latency is.
- The decode split policy was tuned at 262k. At 1M it is smooth rather than pathological, but it
  has not been swept at that length.
- **A 1M boot can fail on the CUDA Graph allowance, transiently.** One 1,048,576-token
  `--tp 2` boot aborted at startup with `CUDA Graph preparation consumed 44433408 bytes on device
  0, exceeding the planned per-device allowance of 20971520 bytes`. The identical command line had
  booted minutes earlier and booted again on the immediate retry, and successful boots of that same
  configuration report only 2.00-3.44 MiB of captured graph memory per device, so the 20 MiB
  per-device allowance is too tight against a capture pool that is not deterministic. The
  workaround is to retry; the fix is to raise the allowance or size it from a measured reservation.
  The failing log is kept at
  `eval/results/cross-engine-nvfp4/ninfer-500w-700000-mtp0-cudagraph-allowance-failure.txt`.
- **Three power conditions, and every number carries one.** The campaign ran at a **400 W** per-GPU
  cap, the publishable subset was re-measured at **575 W**, and the cross-engine comparison against
  vLLM was taken at **500 W**. Never quote a figure without its condition, and never compare figures
  across conditions.
- **What is not measured.** Concurrency C=2 was measured only at the 400 W cap, and the 653k tier
  was not re-measured at 575 W. The soak is greedy only; a seeded temperature > 0 soak has not been
  run. The cross-engine comparison has no vLLM MTP-off row -- speculative decoding is fixed at
  vLLM's launch and that restart was not made -- and the prefill-chunk difference behind vLLM's
  prefill lead (`--prefill-chunk 1024` against `--max-num-batched-tokens 16768`) was not swept on
  either engine. Independent FP64-oracle conformance for shard extents covers the FP8 A8 row
  extents only -- the NVFP4, Q4, Q5, W8 and vocabulary shard geometries are qualified pairwise
  against the `--tp 1` kernel on the whole weight, plus the model-level parity, retrieval and
  MTP-argmax evidence.

This section and [Performance](docs/performance.md) carry the headline figures with their
reproduction commands. The design decisions behind them -- the collective transport, the shard map,
the YaRN constants, and what each correctness gate actually proves -- are in
[Dual-GPU (TP2) execution and YaRN 1M context](docs/maintainer/tp2-yarn-1m.md).

## Capabilities

The five published artifact profiles support:

- text generation with thinking and non-thinking prompt modes;
- image, multi-image, video, and mixed multimodal messages;
- chunked prefill and CUDA Graph decode;
- startup-bounded small-scale concurrent serving with true batched decode;
- MTP speculative decoding with draft windows from one to five;
- BF16 and INT8 group-64 KV cache;
- model- and thinking-mode-aware official sampling defaults, with explicit greedy, temperature,
  top-k, top-p, min-p, and presence/frequency-penalty overrides;
- compatible-prefix reuse;
- OpenAI Responses Core, OpenAI Chat Completions, and Anthropic Messages, including streaming and
  usage accounting;
- prompt-rendered function tools and parsed tool calls.

The 35B-A3B target additionally supports text-only DFlash speculative decoding with draft windows
from one to fifteen.

The local `gguf-q4-k-m` identity supports text and MTP, including the public CLI and HTTP Engine
route. It rejects Vision. The V100X2 acceptance workload uses one active request, native RoPE,
180,000-token capacity, and a maximum MTP draft window of three.

## Current limits

- Only the six `(model_id, weights_id)` artifact identities listed above are accepted product
  identities.
- This checkout targets Volta, Ampere and early Ada. One CUDA device is the generic CLI default;
  the V100X2 launcher selects exactly two with `--tp 2 --devices A,B`.
- One Engine owns one resident model and supports a startup-fixed capacity of 1–8 active requests.
  Decode-ready requests are compacted at round boundaries and executed in one batched model
  traversal.
- NInfer does not provide large-scale or preemptive continuous batching, priority/QoS scheduling,
  CPU/GPU offload, or distributed serving. Multi-GPU execution is exactly the two-device
  tensor-parallel width described above: one process, one resident model, no more than two devices.
  CUDA peer access is used where available, including suitable V100 NVLink topologies; GeForce
  pairs without peer access use the host-staged transport.
- `--max-context` is the logical ceiling of each sequence and is configurable up to the registered
  models' native 262,144-token limit, or up to 1,048,576 tokens under `--rope yarn` on the 27B
  targets. `--kv-capacity N` explicitly sizes the shared Main Text KV
  pool for all active and retained sequences, while `--kv-capacity auto` selects the largest usable
  capacity from the memory remaining after weights are loaded while preserving 1 GiB of sizing
  headroom. Omission defaults to one `--max-context` worth of pages. The resolved pool is fixed at
  startup and is not divided statically among request lanes.
- Tool calls are parsed and returned to the client; NInfer does not execute tools.
- The C++ headers are used by the in-tree applications and are not distributed as an installed SDK.

## Documentation

- [Contributing](CONTRIBUTING.md)
- [Documentation index](docs/README.md)
- [CLI](docs/cli.md)
- [HTTP serving](docs/serving.md)
- [Performance](docs/performance.md)
- [CLI examples](examples/cli/)

## License

NInfer is licensed under the [Apache License 2.0](LICENSE). This fork's modifications are under the
same licence; see [NOTICE](NOTICE) for the attribution required by Apache-2.0 §4(b).

The published artifacts are derived from
[Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B),
[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B), and
[Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B). The Qwen3.6-27B NVFP4 artifact
also uses the fixed packed weights from
[rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm](https://huggingface.co/rdtand/Qwen3.6-27B-PrismaSCOUT-Blackwell-NVFP4-BF16-vllm).
The Qwen3.8-27B NVFP4 artifact also uses the fixed mixed FP8/NVFP4 weights from
[unsloth/Qwen3.8-27B-NVFP4](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). These source
repositories are distributed under Apache-2.0. Vendored dependencies retain their own license files
under `third_party/`.
