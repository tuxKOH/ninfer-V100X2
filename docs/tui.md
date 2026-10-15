# NInfer terminal launcher

Run from a local terminal or SSH session with Python 3.11 and curses (standard library on Linux).
No GUI, pip packages, GPU allocation or driver changes are required to open the launcher.

```bash
bash tools/v100/ninfer-tui.sh
```

The default is **QUASAR NVFP4 / TP2 / MTP3 / 180K**, INT8 KV, prefill chunk 2560,
CUDA Graphs and prefix reuse enabled, 65536 default output tokens, loopback port 8080,
one active request. It selects the existing `build-v100/apps/ninfer-serve`, or the local
`build-v100-tp4/apps/ninfer-serve` when the first is absent. This build-directory name does
not select TP4. Artifact paths are explicit; models are never downloaded or converted here.

## Controls / 操作

| Key | Action |
|---|---|
| ↑ / ↓ / Tab | select a setting / 选择参数 |
| Enter / Space | edit a value or cycle a switch / 编辑或切换 |
| Ctrl-U while editing | clear the input / 清空输入 |
| `p` | select a preset / 选择预设 |
| `m` | choose an existing known artifact or type an explicit path / 选择模型 |
| `c` | view the exact command / 查看完整命令 |
| `s` | start or stop this launcher's server / 启动或停止 |
| `q` / Esc / Ctrl-C | stop the owned server and exit / 停止自己的服务并退出 |

Use a terminal at least **80 columns × 18 rows**. Settings are locked while the child runs;
stop it before changing the model, GPUs or capacities. CUDA IDs are logical: with
`CUDA_VISIBLE_DEVICES=4,6`, enter `0,1`. The API output limit is still clamped to remaining
context. The model's embedded template determines supported thinking effort; this launcher
sets only the server's default thinking on/off, not a universal `xhigh` request setting.
The `Admission timeout ms` setting maps to `--pending-timeout-ms`; it defaults to `0`, which
disables the admission deadline so a request waits behind a long prefill instead of failing with
a `request_queue_timeout` (the Engine admits at most one prefill owner at a time).

## Presets

| Preset | Configuration |
|---|---|
| `quasar-mtp3` | recommended TP2 / MTP3 / 180000 / chunk 2560 |
| `quasar-mtp4` | default attention TP2 / MTP4 / 180000 |
| `quasar-long` | experimental long attention TP2 / MTP4 / 180000 |
| `quasar-dflash7` | TP2 / DFlash7 / 98304 / chunk 1024 / full proposal head |
| `nvfp4-tp4` | official NVFP4 / four cards / MTP3 / 180000 |
| `fp8-tp4` | native block-128 FP8 / four cards / MTP3 / 180000 |
| `q4-mtp3` | GGUF-derived Q4_K_M / TP2 / MTP3 / 180000 / chunk 4096 |

`quasar-long` retains all historical KV but changes reductions and can change output;
it cannot be combined with DFlash. The launcher leaves it off in other presets.
Both MTP4 presets select greedy sampling and disable prefix reuse to match the coding
measurement's settings; MTP3 retains model sampling defaults and enables prefix reuse.
Relative edited paths are resolved from the repository root, matching the child working directory.
The [QUASAR measurements](performance.md) describe its 85K coding workload and quality limits.
TP4 needs four suitable devices and the corresponding artifact; see [TP4/NVLink](tp4.md).
The C++ server remains the authority for artifact identity, admission and available GPU memory.

Explicit paths and command-only mode:

```bash
bash tools/v100/ninfer-tui.sh --profile quasar-long \
  --binary /home/z/ninfer-V100X2/build-v100-tp4/apps/ninfer-serve \
  --artifact /Models/ninfer-V100X2/quasar-v3/qwen3_8_27b_nvfp4.ninfer

bash tools/v100/ninfer-tui.sh --profile quasar-dflash7 --print-command
```

The UI shows child stdout/stderr, loading/listening state, `/health`, and **one-second Engine
prefill/committed-decode intervals**, not request-phase average rates. Full-precision request
records append to the editable log path (default `profiles/tui-server.jsonl`). It sends SIGTERM
only to its own child, then SIGKILL after five seconds if necessary. External API processes
are untouched; the terminal launcher does not run a background system service.
