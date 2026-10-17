# Alternative models and experimental Vision

QUASAR is the recommended text model. This page retains Q4_K_M conversion and the opt-in TP2
Vision route (official NVFP4, or QUASAR v3 with `--vision`). These profiles have their own numerical
and memory constraints.

TP2 Vision is experimental and disabled by default. The visual-prefix bridge test fails on
the local host because cached and cold prefill produce different greedy continuations; the
cause remains unresolved. Use cold requests with `--no-prefix-reuse` below. QUASAR rejects
Vision, and TP4 Vision / DFlash plus Vision are unsupported.

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

### Experimental TP2 Vision

Use the official NVFP4 artifact, or QUASAR v3 at TP2 (both carry execution-qualified Vision;
GGUF-derived Q4_K_M does not). Start with a small text/KV capacity and visual-token budget on the
two 16 GB cards; the command below is not a measured maximum-capacity profile. `--vision-max-tokens`
bounds merged visual tokens, while the complete text-plus-media prompt must also fit
`--max-context`.

```bash
env LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  build-v100/apps/ninfer-serve /Models/ninfer-V100X2/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 8080 --tp 2 --devices 0,1 --max-concurrency 1 \
  --max-context 4096 --kv-capacity 4096 --kv-dtype int8 --prefill-chunk 1024 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --vision --vision-max-tokens 1024 --no-prefix-reuse --default-max-tokens 512 --no-thinking
```

This disables Engine prefix reuse for all requests on this server, including text. It does not
disable the separate immutable-media preprocessing cache. The recommended QUASAR text profile
keeps prefix reuse. See [multimodal requests](serving.md#multimodal-request) and
[known Vision test limitation](../tests/README.md#experimental-tp2-vision-status).
