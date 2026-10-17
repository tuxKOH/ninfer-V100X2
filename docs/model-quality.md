# QUASAR source-checkpoint quality

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

**Vision is opt-in and experimental for QUASAR.** The source checkpoint and published container
include Vision, and the v3 container stores the complete set in the formats the Vision binder
expects, so `--vision` at TP2 binds and executes it. The default profile remains text-only and
uploads zero Vision weights. As with the official artifact, TP2 Vision is not fully qualified on
this host and its visual-prefix bridge test still fails; the recommended experimental Vision
profile uses `--no-prefix-reuse`. MTP3 uses 8.66 GiB of weights per card; the full artifact occupies 18.42 GiB on disk.
Official NVFP4 and GGUF-derived Q4_K_M remain supported alternatives.
