# NInfer V100X2

[English](README.md) · [详细评测方法](docs/performance.md) · [HTTP API](docs/serving.md)

原生 RAM-KV 实验入口：`--ram-kv-window N`，支持 27B SM70、TP1/TP2、单请求 Text/MTP。
首版采用免训练 **token 匹配检索**，并非 KVMem 的 Q/K 向量检索移植。RAM 无损存储 packed KV，
但未选中的历史页不参与注意力，长 prefill 也会淘汰，可能漏召回；默认完整历史路径不变。
详见 [存储与状态合同](docs/maintainer/paged-kv-cache.md#15-optional-native-ram-kv-experiment) 和
[启动选项](docs/cli.md)。外部 KVMem 成绩不能当作这条原生路径的性能数据。

性能数据分为 [P2P 开启](#性能p2p-开启) 和 [P2P 关闭](#性能p2p-关闭) 两区；
构建、模型准备和 [API 启动命令](#cli-和-api-启动) 在下方统一说明。

面向 **两张 Tesla V100-SXM2 16 GB、CUDA 12.8、SM70** 的 Qwen3.8-27B 高速推理分支，
重点优化单个活跃请求。由上游 RTX 3060 TP2 路线与 `geoffwatts/ninfer-v100` 的 Volta
实现结合而来。本文介绍本分支的改动和实测；外部源模型质量评测单独标明，
不把上游 RTX 5090 等平台的结果当作 V100 成绩。

采用 TP2 **分张量**执行，不是 32/32 层的分层卸载。**纯文本推理主推 QUASAR NVFP4 v3**，
通用任务用 180000-token 容量、完整 INT8 group-64 KV、CUDA Graph 和 MTP3；高接受率
结构化输出可选 98304 容量的 DFlash7。官方 NVFP4 和 GGUF 衍生 Q4_K_M 仍可使用。
启动脚本未传模型时仍默认 Q4_K_M；下方[推荐启动命令](#cli-和-api-启动)显式选择 QUASAR。
“MTP 3-0”在这里是最多提出三个草稿 token、允许接受零个，不保证每轮接受三个。

**容量不等于实际上下文占用。** 下表中的 85K 是实际输入 85000 token；180K 是容量上限。
Decode 只统计已提交输出，不把被拒绝的草稿算入吞吐。NVFP4 和 Q4_K_M 使用不同量化权重，
这些表不是两种量化的质量等价证明。

## 主推模型：QUASAR NVFP4

使用 [QUASAR NInfer v3 成品](https://huggingface.co/MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer/tree/v3)。
本机[实测](#quasar-nvfp4) MTP3 在 3072-token 代码输入下达到 **139.80 tok/s**，
85K-token 代码输入热请求为 **100.34 tok/s**，均为已提交 wall decode。
DFlash7 在原生 32 条 JSONL 任务达到 **249.00 tok/s**，85K 输入为 **153.12 tok/s**。
这些是具体任务成绩，DFlash 并非所有任务都比 MTP 快。

维护者个人实际体验是跟 FP8 一档的。

公开的**源模型质量评测**显示多项任务分数接近 BF16，但不等于数学无损。
[QUASAR 作者模型卡](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4#quality-and-size-comparison)
报告 GPQA-D 两轮共 396 个答案、AIME'26 三轮共 90 个答案：

| 源模型 | GPQA-D（%） | AIME'26（%） |
|---|---:|---:|
| BF16 原模型 | 91.41 | 100.0 |
| QUASAR NVFP4 | 90.91 | 100.0 |

[Rieker 的独立对比](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/192)采用 DGX Spark GB10、
vLLM、FP8 KV 和 MTP5，不是本机 V100 路线。MBPP/HumanEval/GSM8K 关闭 thinking，
分别测 257/164/500 题；PPL 使用 323 个窗口，文本 KLD 使用 49 条提示：

| 源模型 | MBPP（%） | HumanEval（%） | GSM8K（%） | PPL ↓ | 文本 KLD ↓ |
|---|---:|---:|---:|---:|---:|
| BF16 原模型 | 70.8 | 93.3 | 97.0 | 7.993 | 参照 |
| 官方 FP8 | 69.3 | 95.1 | 97.4 | 8.029 | 0.0117 |
| QUASAR NVFP4 | 68.9 | 93.9 | 96.8 | 8.247 | 0.0682 |

任务准确率接近 BF16，分布保真度则不等同于 FP8。两项公开评测均不代表本分支转换后
`.ninfer` 成品的 V100 质量实测；我们的推理检查也不等于全面质量评分。V100 反量化执行，
没有开启 A4 激活量化；v3 是容器升级，不是一次新的 QAT。

**限制：本分支 QUASAR 不支持视觉。** 源模型和公开容器包含视觉数据，但当前运行时拒绝
QUASAR 图像／视频请求，两卡均不上传视觉权重。MTP3 每卡权重为 8.66 GiB，完整模型文件
占磁盘 18.42 GiB。

TP2 Vision 是 [PR #1](https://github.com/tuxKOH/ninfer-V100X2/pull/1) 引入的**实验性、可选功能**，
默认关闭，不会为 QUASAR 开启视觉。本机视觉前缀 bridge 测试未通过：缓存与冷 prefill
产生不同的 greedy 输出，原因仍未确定。建议采用[禁用前缀复用的视觉 API 配置](#实验性-tp2-视觉)，
避开尚未通过验证的视觉缓存路径。视觉需要额外显存；TP4 Vision、DFlash 加视觉仍不支持。

## 本分支的改动

- Q4_K_M prefill：协作解码 GGML-K 块，写入调用方 FP16 workspace，再交给 Volta CUTLASS
  Tensor Core GEMM；GDN 控制投影保留 FP32 输出。源 GGUF 的 Q4_K/Q6_K 编码和 scale 不变。
- NVFP4/FP8：宽 prefill 每次调用只解码一次权重，支持 TP2 预打包矩阵；窄投影走 QPN
  Tensor Core 路线。T≥2048 的宽 TP2 NVFP4 MLP 使用 256-column GEMM tile。
  SwiGLU 的 gate/up 中间值保留 FP32，仅最终激活写回 BF16。
  修复了预打包 FP8 词表头在单 token 和 chunk 尾部错误走 row-major GEMV 的问题。
- 长上下文注意力：共享 QK 分块和 softmax 权重，协作读取 split 向量，保留原有 FP32 累加顺序。
  不裁剪历史，不用检索子集代替完整注意力，不额外降低 KV 精度。
- DFlash 局部注意力：V100 在分片数量足够时采用每行一个 warp，并以 FP32 保存 split
  numerator；极短窗口仍走 CTA 路径。支持 2048/4096 窗口，主模型完整历史 KV 不变。
- GDN prefill：Q/K 每行只做一次 FP32 归一化，供状态 tile 复用；FP32 递推顺序不变。
  3072-token TP2 的 GDN scratch 为 24 MiB。
- TP2 通信：图可捕获的 UVA D2D copy 加双向事件顺序；启动时检查双向精确传输。
  支持 direct P2P 和经过验证的 CUDA-managed staging，不另设显式 pinned-host 通信分支。
  PCIe 对卡仍受 translated IOMMU 限制；实际启用的 NVLink mesh 通过 NVML 远端 PCI
  端点确认后单独验证，不再被 PCIe 的 IOMMU 防护误禁用，也不修改系统配置。
  已验证 P2P 的 SM70 小消息（≤80 KiB）直接读取对卡并求和到本卡 staging，
  两卡读完才覆盖原输入；宽 prefill 和 P2P 关闭时继续走 DMA。
- Volta 残差路径：Q5 与 BF16 TP2 row-shard 使用可在 SM70 执行的 kernel，并通过独立数值检查。
- NVFP4 v3：读取官方 `NINFER\x00\x03` 单文件容器，映射到已注册的
  `qwen3.8-27b/nvfp4` 身份，不重打包权重字节；使用容器原始 chat template。
  容器可选的五层 DFlash2 组件已绑定为 text-only TP2 实验路径。
- MTP / DFlash 双卡词表传输：greedy target 验证只交换各分片的 argmax 值和 ID；
  DFlash 草稿 selector 合并各卡的精确 top-16，不搬运完整词表 logits。
  采样或混合 batch 的验证仍保留完整 logits 路径。
  V100 的窄 BF16 草稿投影直接读取 BF16，以 FP32 SIMT 累加，避免小矩阵的 CUTLASS 开销。
- 前缀复用：TP2 与 optimized MTP3 可以恢复当前前沿或完整 turn/response checkpoint，
  只计算后续新增提示；支持重复恢复、精确命中和提前停止后的继续生成。

## NVLink 远端部署：TP2 阶段

2026-10-02 在朋友的四卡机器上实测：4×V100-SXM2 16 GB，全部两两 NV2，300 W/卡，
驱动 580.178.04，CUDA 12.8 / SM70。**下表 NInfer 只使用 0,1 两张卡，TP2；另外两卡空闲。**
真实 NVLink P2P 在主机 IOMMU 为 DMA-FQ 时仍通过验证，本次未修改 GRUB、驱动或重启。
它不是本机 PCIe P2P 的同硬件 A/B 对照。

官方 Qwen3.8-27B NVFP4 v3、完整 INT8 group-64 KV、180000 容量（分配 180032），
prefill chunk=2560、greedy optimized MTP3、CUDA Graph。每档两轮冷提示，无前缀复用、
无额外请求 warmup；加载及图预热不计入请求。每轮输出 1 个 prefill token＋512 个计时 decode token，
保存的输入保留最后的代码任务和 assistant 后缀。均值 ± 样本标准差：

| 实际输入 tokens | Prefill tok/s | 已提交 decode tok/s | Wall decode tok/s | MTP 接受率 |
|---:|---:|---:|---:|---:|
| 3072 | 2101.49 ± 4.03 | 111.19 ± 0.01 | 111.10 ± 0.01 | 72.16% |
| 8192 | 2114.14 ± 6.82 | 109.04 ± 0.03 | 108.91 ± 0.03 | 70.67% |
| 16384 | 2050.33 ± 1.22 | 104.46 ± 0.02 | 104.34 ± 0.02 | 69.00% |
| 32768 | 1887.71 ± 0.55 | 101.99 ± 0.07 | 101.89 ± 0.08 | 74.47% |
| 65536 | 1604.61 ± 3.27 | 90.28 ± 0.03 | 90.23 ± 0.03 | 76.39% |
| 85000 | 1463.73 ± 4.35 | 85.53 ± 0.04 | 85.50 ± 0.05 | 77.11% |

每档重复输出的 513 个 token ID 及草稿统计全部一致，计时窗口没有 EOS/EOG。
这只是固定代码输出窗口的性能测量，不是完整代码评分。
远端报告位于 `results/tp2-nvlink/`；[bench_tp.py](tools/v100/bench_tp.py) 通过公开
`ninfer_bench` 顺序执行实际占用、容量和无 MTP 对照矩阵。

容量对照固定相同的 **512 输入**、chunk=1024、MTP3，每档两轮、**256 个计时 decode token**。
1024 容量放不下 512 输入再加 512 个计时输出，因此全档统一使用 256。各档实际 KV 分配
等于标称容量，重复的 257 个输出 ID 与草稿统计均一致，接受率均为 70.33%。这不是长输入测试。

| 最大上下文容量 | Prefill tok/s | 已提交 decode tok/s | Wall decode tok/s |
|---:|---:|---:|---:|
| 1024 | 1669.31 ± 29.44 | 112.60 ± 0.04 | 112.56 ± 0.03 |
| 2048 | 1676.79 ± 30.19 | 112.86 ± 0.01 | 112.82 ± 0.01 |
| 4096 | 1666.35 ± 33.44 | 112.70 ± 0.02 | 112.66 ± 0.02 |
| 8192 | 1670.13 ± 31.18 | 112.71 ± 0.02 | 112.67 ± 0.02 |
| 16384 | 1669.02 ± 34.89 | 112.91 ± 0.03 | 112.86 ± 0.03 |
| 32768 | 1668.32 ± 37.79 | 112.81 ± 0.04 | 112.75 ± 0.03 |
| 65536 | 1670.67 ± 27.88 | 112.97 ± 0.01 | 112.90 ± 0.02 |

无草稿对照使用相同保存输入、180000 容量、chunk=2560、512 个计时输出，
改用 `--spec none --draft-tokens 0`，不加载 optimized draft head：

| 实际输入 | 无 MTP prefill tok/s | 无 MTP decode tok/s | MTP3 decode tok/s |
|---:|---:|---:|---:|
| 3072 | 2132.37 ± 8.53 | 39.584 ± 0.004 | 111.19 ± 0.01 |
| 85000 | 1480.16 ± 4.93 | 28.349 ± 0.003 | 85.53 ± 0.04 |

各对照内部重复输出一致；85K 的 MTP/无 MTP 输出一致，3K 则不一致，不能把速度倍率
当作普遍 token 一致或不降智的证明。[分阶段耗时](docs/performance.md#separate-nvlink-deployment-tp2-stage)
另列加载、准备、prefill 与 decode。

### NVLink 部署：NInfer TP4

2026-10-03 在上述四卡 NV2 主机实测，四张卡全部参与计算。官方 Qwen3.8-27B NVFP4 v3，
180000 容量（分配 180032）、INT8 group-64 KV、chunk 2560、greedy 优化草稿头 MTP3、
CUDA Graphs。每点两次冷请求，无前缀复用或额外 warmup；输出为一个 prefill token 加
512 个已提交计时 decode token。加载/图预热不计入请求耗时。均值 ± 样本标准差：

| 实际输入 token | Prefill tok/s | 已提交 decode tok/s | Wall decode tok/s | MTP 接受率 |
|---:|---:|---:|---:|---:|
| 3072 | 2381.83 ± 37.81 | 125.68 ± 0.08 | 125.42 ± 0.08 | 73.13% |
| 8192 | 2415.01 ± 15.32 | 136.81 ± 0.07 | 136.61 ± 0.07 | 80.62% |
| 16384 | 2381.76 ± 9.29 | 124.09 ± 0.08 | 123.82 ± 0.09 | 72.31% |
| 32768 | 2279.47 ± 2.72 | 117.96 ± 0.02 | 117.73 ± 0.02 | 74.47% |
| 65536 | 2072.25 ± 0.66 | 109.12 ± 0.13 | 108.99 ± 0.13 | 79.30% |
| 85000 | 1957.93 ± 2.16 | 102.60 ± 0.08 | 102.51 ± 0.08 | 78.56% |

全部重复运行的 513 个输出 ID 和草稿计数一致，无 EOS/EOG。85K 比此前同机 TP2 快照
的 prefill/decode 分别提升 33.76% / 19.95%。接受率和运算分组不同，因此这不是只改变
TP 宽度的严格 A/B，也不代表不同宽度的输出逐 token 相同。远端报告位于
`results/tp4-nvfp4/`；固定窗口吞吐不是完整代码任务的质量评分。

四卡 NVFP4 graph/eager、采样切换、B=2 MTP 和全部 64 个严格 teacher-forced 输出位置
通过，argmax 零分歧。缓存 checkpoint、连续追加、exact frontier、轮内截断均通过精确
状态对照。缓存/冷启动在第 28 个输出位置的一处分歧是 BF16 并列最大值，logit 差距为 0；
stop 截断与输出预算截断随后追加的输出、logits 和接受计数完全一致。不宣称冷启动重新
prefill 与保留状态计算逐位相同。

### 四卡 1Cat/vLLM 对照

朋友机器上的隔离启动脚本位于 `/home/z/1cat-vllm-v100`，使用全部四张 NV2 V100-SXM2，
复用已有 OneCat Studio 模型目录，监听 loopback 6100/6101；原有 8888 UI 没有改动。
180000 最大容量、MTP3、1570-token 短提示、流式输出 256 token 的实际结果：

| 1Cat 模型 | TTFT | 流式 decode |
|---|---:|---:|
| Qwen3.8-27B NVFP4 | 0.955 s | 83.03 tok/s |
| Qwen3.8-27B FP8 | 1.590 s | 83.23 tok/s |

这是短提示单请求 HTTP 测试，不是 85K 占用上下文，也不是 NInfer TP4 成绩。脚本调用模型
自带的 FP8/NVFP4 kernel；NInfer 原生 block-scaled FP8 已单独完成测评。对照完成后已停止
1Cat 进程，并启动 NInfer TP4 API 到 loopback 6200。

远端仓库内的 `bash tools/v100/serve-tp4.sh /home/z/Models/ninfer/qwen3_8_27b_nvfp4.ninfer`
启动四卡 loopback `6200` API：180K 容量、MTP3、单请求、默认最大输出 65536、前缀复用开启。
文本 OpenAI/Responses/Anthropic 契约、不支持图像的拒绝以及重复提示恢复 turn checkpoint 均已验证。
通过 SSH 隧道访问，API 没有直接暴露到公网：

```bash
ssh -N -L 6200:127.0.0.1:6200 -p 12031 z@你的主机地址
curl http://127.0.0.1:6200/health
```

图形启动器运行 `bash tools/v100/ninfer-gui.sh`。PyQt 控制台可调 artifact、TP/设备、
上下文/KV 容量、prefill chunk、MTP/DFlash 草稿窗口、CUDA Graph/前缀缓存、并发、输出
上限和采样默认值，并在启动前拦截不支持的组合。详见 [`docs/gui.md`](docs/gui.md)。

**NVFP4 与原生 block-128 FP8 的 TP4 整模型/状态门、前缀复用、容量/普通解码对照和
TP4 HTTP 部署均已完成。**

原生 FP8 TP4 实际输入阶梯（同为 180K 容量、INT8 KV、MTP3、两次冷请求、512 个已提交
decode token）如下：

| 实际输入 | Prefill tok/s | 已提交 Decode tok/s | Wall decode tok/s | 接受率 |
|---:|---:|---:|---:|---:|
| 3072 | 2415.00 ± 13.42 | 98.75 ± 0.02 | 98.59 ± 0.02 | 80.04% |
| 8192 | 2445.95 ± 5.48 | 100.16 ± 0.01 | 100.07 ± 0.02 | 80.44% |
| 16384 | 2404.93 ± 4.30 | 95.48 ± 0.01 | 95.31 ± 0.02 | 77.27% |
| 32768 | 2298.05 ± 0.04 | 89.15 ± 0.01 | 89.02 ± 0.01 | 74.68% |
| 65536 | 2089.18 ± 1.61 | 85.05 ± 0.11 | 84.99 ± 0.11 | 80.22% |
| 85000 | 1972.43 ± 2.89 | 81.53 ± 0.09 | 81.47 ± 0.09 | 80.58% |

原生 FP8 容量阶梯（512 输入、256 输出、chunk 1024）完整数据如下：

| 容量 | Prefill tok/s | Decode phase tok/s | Decode wall tok/s |
|---:|---:|---:|---:|
| 1024 | 2003.40 ± 38.05 | 101.52 ± 0.06 | 101.49 ± 0.07 |
| 2048 | 1996.61 ± 50.16 | 101.43 ± 0.08 | 101.40 ± 0.08 |
| 4096 | 1997.52 ± 47.56 | 101.48 ± 0.01 | 101.45 ± 0.01 |
| 8192 | 1986.08 ± 56.70 | 101.12 ± 0.05 | 101.10 ± 0.05 |
| 16384 | 1990.54 ± 54.50 | 101.55 ± 0.01 | 101.52 ± 0.01 |
| 32768 | 2000.66 ± 49.85 | 101.33 ± 0.20 | 101.29 ± 0.20 |
| 65536 | 1999.82 ± 41.27 | 101.26 ± 0.06 | 101.20 ± 0.06 |

所有容量分配、输出 ID 和草稿计数均重复一致。
无 MTP 对照为 3K 40.98、85K 33.12 tok/s。FP8 teacher-force 门记录 3 个草稿 argmax
差异，最大 emitted-logit deficit 0.125；ordinary 冷重算为 1 个、同为 0.125。这是文档中
规定的数值包络，不是逐 token 完全无损声明。

同一台四卡机器已通过生产通信集（直连 NVLink 与 staged fallback、非均匀 gather、broadcast、
图重放）以及 BF16 四分片投影/恰好一次 residual 测试：10 个几何/阶段组合，eager 与 CUDA
Graph replay 均零失败。四卡 NVFP4/row-FP8/原生分块 FP8 注意力与 GDN 投影，以及
T=1/4/128/2560 的 BF16 四分片 GDN 控制，已通过独立 FP64 检查。四卡 NVFP4 普通 eager
短请求已正常输出代码；这只是冒烟检查，不是吞吐或质量评分。新增原生分块 FP8 的 Linear、
融合 SwiGLU/残差独立 FP64 oracle、TP4 精确字节分片、6Q/1KV 注意力、4QK/12V GDN replay
已在本机 SM70 通过；这些是算子验证，不是四卡整模型成绩。

Q4 优化草稿头的四分片形状已补齐并通过 FP64 检查。并行草稿选词现在只合并每卡
FP32 最大值和 I32 token ID，不再收集完整草稿词表；采样仍使用完整 target logits。
修改后的本机 TP2 MTP graph/eager、采样切换、B=2 状态与全部 64 个 teacher-forced
输出位置检查通过，argmax 零分歧。四卡图内存已采用独立启动预算，并已通过真实四卡
NVFP4 MTP 启动/状态检查。

朋友机器现有 OneCat FP8 源已转换为独立 `qwen3.8-27b/fp8` 产物：31.30GB（十进制），
保留 E4M3 权重和全部 128×128 BF16 乘数，不重新量化；embedding、输出头和 MTP stem
保持 BF16。它与 NVFP4 包中的 row-scaled FP8 不是同一种格式。
[原生 FP8 产物说明](docs/maintainer/qwen3.8-27b-artifact.md#native-block-128-fp8-textmtp-package)。

TP4 当前选择 SM70 的 Qwen3.8-27B NVFP4/原生 FP8 Text/MTP；原生 FP8 要求 TP4，
四卡均不开放 Vision/DFlash。上述 graph/eager、MTP 提交、前缀缓存、HTTP 和上下文阶梯
已在独立四卡机器上验证；上面的 TP2、1Cat 数据不会改名成 TP4 成绩。

## 性能：P2P 开启

2026-10-01 实测，双卡各 300 W，PCIe 3.0 ×16、PHB 拓扑。重启后使用 `iommu=pt`，
两 GPU 的 IOMMU domain 均为 `identity`，NVIDIA P2P read/write 检查通过，
NInfer 自动启用已有 direct P2P 路径。**这是 PCIe P2P，不是 NVLink。**
该配置在本机通过验证，不代表所有主板或转接方案都能启用 P2P。
两区使用同一套启动命令；NInfer 在启动时依据实际双向通信检查自动选择路径。
`iommu=pt` 本身不是 P2P 成功证明，`nvidia-smi topo -m` 的 PHB 标签也不是。

### QUASAR NVFP4

主推的 `qwen3.8-27b/quasar-nvfp4` 配置用公开 Engine 加载
[QUASAR NInfer v3 成品](https://huggingface.co/MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer)。
开放 SM70 TP2 Text/普通解码/MTP/DFlash；启动脚本的历史默认不变，推荐显式选用 QUASAR。
保留原始 codes/scales 和全部 256 对激活缩放参数。QUASAR 是 QAT，不是数学上的无损压缩；成品转换也含 BF16/W8
精度边界，详见 [模型存储合同](docs/maintainer/qwen3.8-27b-artifact.md#13-quasar-trial-artifact)。

Attention/GDN 输入与输出权重现已和 MLP 一样，加载时进入 Volta QPN 布局；窄输入直接
写最终输出，宽输入每次调用只反量化一次权重供 CUTLASS 使用，再分发 BF16 结果。
不改变文件中的权重／divisor，没有开启 A4 激活量化或裁剪历史注意力。

2026-10-03 本机双 V100：PCIe P2P、**无 NVLink**、各 300 W、CUDA 12.8；保存的代码
输入，chunk 2560、完整 INT8 group-64 KV、greedy optimized-head MTP3、CUDA Graph。
每行测两次，无前缀复用；3072 输入无额外预热，85000 输入先完整预热一次。每次输出
一个 prefill token，加 **512 个已提交 decode token**。下表 decode 用提交数量除以首
token 后的请求 wall 时间，不计加载／图预热；均值 ± 样本标准差。

| 路径 | 实际输入 / 容量 | Prefill tok/s | 已提交 wall decode tok/s | 接受率 |
|---|---:|---:|---:|---:|
| 接入 QPN 输入路径前 | 3072 / 8192 | 376.71 ± 0.77 | 102.02 ± 0.06 | 85.78% |
| QPN / MTP3 | 3072 / 8192 | 1751.55 ± 30.90 | 139.80 ± 0.08 | 85.19% |
| QPN / MTP3，热请求 | 85000 / 180000 | 1315.28 ± 2.61 | 100.34 ± 0.05 | 86.62% |

短输入配对收益为 **prefill 4.65 倍、decode 1.37 倍**，接受率反而略降。85K 热请求平均
prefill/decode 阶段／完整请求为 64.625/5.100/69.729 秒；Engine decode 阶段吞吐为
100.40 tok/s。MTP 每卡权重仍为 **8.66 GiB**。
各路径两次输出和草稿计数一致；新旧短输入路径从第 270 个输出 token 起不同，数值路线
变化不代表逐 token 保持，也不能当作不降智证明。独立 FP64 输入投影检查覆盖实际 TP1/TP2
形状及 QPN/CUTLASS 边界；公开 Engine 的 MTP、DFlash 各通过 64 个严格 teacher-forced
位置及 graph/eager 采样检查。尚未做 BF16 质量评分或满占用 180K/256K 测量。
本轮报告位于 `profiles/bench/quasar-qpn-projections/`，初始未优化对照保留在
`profiles/bench/quasar-v3-tp2-comparison/`。

同机 **QUASAR DFlash7 JSONL**：容量 98304、chunk 1024、完整 INT8 KV、full proposal head、
greedy CUDA Graph，无前缀复用。复用此前的 32 条 JSONL 任务和不同技术文档背景；每行先
完整预热一次，再测两次。启用模型默认停止；六次答案各发布 1190 token（扣首 token 后
计时 1189），全部自然结束，记录内容／算术／字段顺序检查全过。每行两次输出 ID 和
草稿计数一致。

| JSONL 实际输入 token | Prefill tok/s | 已提交 wall decode tok/s | DFlash 接受率 |
|---:|---:|---:|---:|
| 118（原生提示） | 564.22 ± 0.33 | 249.00 ± 0.08 | 99.05% |
| 4096 | 1577.84 ± 1.11 | 235.97 ± 0.04 | 99.05% |
| 85000 | 1184.19 ± 1.07 | 153.12 ± 0.02 | 98.30% |

85K JSONL 平均 prefill/decode wall／完整请求为 **71.779/7.765/79.546 秒**。这是高接受率
结构化输出成绩，不代表普通代码／故事也能达到 249 tok/s；上面的 MTP 代码表是另一种
任务，不能拿来算 DFlash 的配对增益。保留完整历史注意力，没有用近似 RAM-KV 提速。
报告：`profiles/bench/quasar-qpn-projections/jsonl-dflash7.json`；
[复测工具与精确答案检查](bench/README.md#stop-aware-v100-task-probes)。

**QUASAR 完整任务矩阵**：同一份保存的输入，容量 98304、chunk 1024、完整 INT8 KV；
每项先完整预热一次，比较 optimized-head MTP3 与 full-head DFlash7。80 次正式测量
均在 2048 输出上限前自然停止，无前缀复用。下表 **D/M = DFlash7 / MTP3 已提交 wall
decode tok/s**；原生提示为两次均值，其余每个任务／长度／后端仅测一次，不是稳定均值。

| 实际输入 token | 中文故事 D/M | 英译中 D/M | 32 条 JSONL D/M | 逻辑题 D/M |
|---:|---:|---:|---:|---:|
| 原生：129 / 395 / 118 / 417 | 56.45 / 89.58 | 162.90 / 143.66 | 247.73 / 163.33 | 204.68 / 152.93 |
| 1024 | 59.98 / 89.34 | 154.45 / 143.26 | 242.15 / 161.98 | 185.00 / 143.35 |
| 2048 | 58.38 / 85.06 | 149.46 / 140.75 | 232.84 / 158.92 | 193.55 / 151.54 |
| 4096 | 53.75 / 83.76 | 144.89 / 138.87 | 234.86 / 158.91 | 151.70 / 137.03 |
| 8192 | 55.62 / 82.65 | 145.07 / 139.49 | 229.81 / 156.56 | 195.49 / 143.77 |
| 16384 | 58.74 / 85.15 | 124.65 / 129.88 | 221.85 / 151.38 | 177.22 / 139.75 |
| 32768 | 48.32 / 77.06 | 124.26 / 124.26 | 198.65 / 139.45 | 162.39 / 132.71 |
| 65536 | 41.32 / 64.82 | 98.82 / 102.86 | 165.74 / 119.52 | 135.01 / 112.12 |
| 85000 | 39.26 / 61.28 | 87.06 / 93.29 | 152.72 / 111.73 | 127.43 / 105.94 |

85K 的 DFlash JSONL／逻辑 decode 分别快 36.69%／20.28%，但故事、翻译更慢；prefill
也更慢，为 1176.69–1186.82，对比 MTP 的 1226.43–1238.87 tok/s。JSONL 完整请求只从
79.895 秒变为 79.794 秒，逻辑题完整请求反而更慢，不能宣称普遍端到端提速。
所有 JSONL 精确记录／顺序、逻辑映射／CHECK 检查通过；翻译只检查章节和词汇表结构，
未独立评分翻译准确性。故事只在 8K／16K／64K 满足要求的字数范围。
**36 组配对有 31 组完整输出 IDs 一致**，85K 四组均一致，不代表通用逐 token 一致或无损。
本轮遭遇意外整机重启，已保留完成项并只补缺失项；最后续跑采用约十分钟负载／三分钟
休息，不能据此证明持续负载稳定。
[完整阶段耗时、接受率、输出数及质量限制](docs/performance.md#quasar-v3-stop-aware-task-matrix)。

QUASAR 当前已是纯文本加载：**两张卡均不上传任何视觉对象**，视觉请求会被拒绝。
原模型文件保持不变；文件中仅供验证的视觉数据占磁盘，不占 GPU 权重显存。
因此物理删掉这部分不会继续降低当前纯文本配置的显存，也不能解释速度提升。

模型保存在 `/Models/ninfer-V100X2/quasar-v3/qwen3_8_27b_nvfp4.ninfer`，18.42 GiB，
已核对发布者 SHA256。本次编译目录为 `build-v100-tp4`，下面实际使用的仍是 **TP2**：

```bash
LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64" \
  build-v100-tp4/apps/ninfer /Models/ninfer-V100X2/quasar-v3/qwen3_8_27b_nvfp4.ninfer \
  --tp 2 --devices 0,1 --max-context 8192 --kv-dtype int8 --prefill-chunk 2560 \
  --spec mtp --draft-tokens 3 --lm-head-draft --no-thinking --greedy \
  --max-new 512 --prompt '用 C++ 编写一个有界阻塞队列。'
```

DFlash 改用 `--max-context 98304 --prefill-chunk 1024 --spec dflash --draft-tokens 7`，
去掉 `--lm-head-draft`，即可匹配上面 full-head 的测量配置。

### NVFP4 v3：实际上下文占用

官方 NVFP4 v3、TP2、完整 INT8 group-64 KV、greedy（temperature=0）、MTP3、
optimized draft head、CUDA Graph。容量统一 180000（实际 KV 分配 180032），prefill chunk=3072。
每档两轮冷提示，无前缀复用、无额外请求 warmup；图预热在计时之外。
每轮生成 513 token，其中首个来自 prefill，后续 **512 个是计时 decode 输出**。
吞吐为均值 ± 样本标准差。
此表早于下方 greedy / 小消息通信更新，其他上下文档位尚未按新实现重测。

| 实际输入 token | Prefill tok/s | 已提交 Decode tok/s | MTP 接受率 |
|---:|---:|---:|---:|
| 3072 | 1783.96 ± 43.38 | 105.48 ± 0.01 | 72.97% |
| 8192 | 1754.43 ± 18.24 | 114.49 ± 0.01 | 81.98% |
| 16384 | 1699.06 ± 9.73 | 108.29 ± 0.003 | 79.47% |
| 32768 | 1596.24 ± 2.94 | 101.63 ± 0.03 | 79.69% |
| 65536 | 1397.83 ± 0.72 | 88.34 ± 0.04 | 79.12% |
| 85000 | **1306.48 ± 3.95** | **83.18 ± 0.13** | **79.12%** |

每档两轮输出 IDs 和 MTP 统计一致，无 EOS/EOG，均完成指定窗口。
不同长度的提示保留完整任务及 assistant 后缀，但代码正文、接受率不同，
不能由 8K 档更快推断上下文越长越快。这些是固定窗口吞吐，不是完整代码任务得分。

85K 输入、180K 容量满足此前提出的 1000 prefill / 70 decode tok/s 目标。
同输入关闭 P2P 的控制组为 79.18 decode tok/s，开启后为 83.18，提升 **5.05%**；
两路径各两轮的所有 513 个输出 IDs 和 speculative 字段一致，每轮 accepted/drafted=360/455，
152 rounds。不能拿下方旧语料的 78.424 tok/s 计算严格的 P2P 因果增益。

85K resident 请求平均 prefill 65.061 秒、decode 6.155 秒、总计 71.220 秒。
模型加载 19.662 秒，只在进程启动时进行一次，不计入请求吞吐。
全部上下文的阶段时间、wall decode、接受数量和测量限制见[性能文档](docs/performance.md#p2p-enabled)。

### 合格的 NVFP4 prefill 更新

85K 输入、180K 容量、MTP3、chunk=2560 的同条件对照，各实现测两轮冷请求：
加宽 TP2 MLP tile 后，prefill **1297.05 ± 2.21 → 1311.92 ± 2.06 tok/s（+1.15%）**。
全部 513 个输出 IDs 和接受统计相同；decode 为 81.63 → 81.71 tok/s，不据此宣称稳定提升。
测试时桌面显存占用使更大 chunk 无法通过启动余量检查，因此采用 2560；
上面的旧 3072-chunk 表不是这项改动的配对控制组。独立数值判据未放宽，
本轮没有达到交付条件的新有损注意力入口。

### 精确 greedy 与小消息 P2P 更新

同一份实际 85000-token 输入、180000 容量、chunk=2560、官方 NVFP4 v3、INT8 KV、
greedy optimized MTP3、CUDA Graph，各实现测两轮冷请求：

| 实现 | Prefill tok/s | 已提交 Decode tok/s |
|---|---:|---:|
| 控制组，完整 logits 验证 | 1315.36 ± 2.45 | 81.80 ± 0.08 |
| 精确分片 winner | 1312.31 ± 3.15 | 82.40 ± 0.04 |
| 并行分片 argmax | 1312.38 ± 1.68 | 82.43 ± 0.13 |
| 再加入小消息 direct-peer 求和 | 1311.95 ± 3.70 | **85.21 ± 0.06** |

组合 decode 提升 **4.17%**，其中通信这一步相对上一版提升 3.37%；
单独 argmax 的整段增量在噪声内，不宣称 prefill 提升。
所有轮次的 513 个输出 IDs 和完整接受统计一致：每轮接受 357/463 drafts、155 rounds，
接受率 77.11%。权重、完整历史注意力、KV 精度和舍入边界不变；采样仍保留完整 logits 与原有 penalty 更新。
新实现的 resident 请求平均 prefill 64.789 秒、decode 6.009 秒，总计 70.803 秒。
不能拿上面的旧 chunk=3072 表计算这轮增益。
[阶段耗时和验证限制](docs/performance.md#exact-greedy-and-small-message-p2p-update)。

### NVFP4 v3：最大容量，固定 512-token 输入

固定 512 输入、256 timed decode token（总输出 257）；prefill chunk=1024，其他推理配置不变。
每档丢弃一轮 warmup，再测三轮。这里 decode 使用**首 token 到请求结束的 wall time**，
而上一张表使用 Engine decode 阶段时间。

| 最大容量 | Wall decode tok/s | Prefill tok/s | MTP 接受率 |
|---:|---:|---:|---:|
| 1024 | 106.12 ± 0.01 | 1443.78 | 70.04% |
| 2048 | 106.12 ± 0.04 | 1438.44 | 70.04% |
| 4096 | 105.96 ± 0.02 | 1440.81 | 70.04% |
| 8192 | 105.98 ± 0.01 | 1442.96 | 70.04% |
| 16384 | 106.06 ± 0.04 | 1443.37 | 70.04% |
| 32768 | 106.00 ± 0.03 | 1443.77 | 70.04% |
| 65536 | 106.01 ± 0.06 | 1443.42 | 70.04% |

七档容量均精确分配；21 轮输出和 MTP 统计一致，每轮接受 173/247 drafts，83 rounds。
这是只有 512-token 实际占用的容量测试，不能将约 106 tok/s 当成 64K 实际占用的速度。

### 通信与推理回归

10 KiB BF16 allreduce，500 次 host-sync 测量：

| 路径 | 平均延迟 µs | p50 µs | p99 µs |
|---|---:|---:|---:|
| PCIe P2P 开启 | 26.6085 | 25.901 | 43.910 |
| P2P 关闭，verified staging | 47.7961 | 47.181 | 62.424 |

通信平均延迟降低 44.33%，**不等于整段推理快 44.33%**。两路径的精确传输、
不整齐 shape、guard 和 64 连续轮检查通过。

Q4_K_M 与 NVFP4 v3 的真实**纯文本** MTP、前缀缓存回归通过：Graph/eager 输出、logits、接受状态和
复用前沿一致；两个 probe 的 MTP/plain 32-token 序列一致；各检查 64 个 teacher-forcing
位置，无分歧、worst emitted-logit deficit=0。前缀覆盖重复恢复、追加、前缀变更、
精确前沿、接受轮内提前停止、改写回复后的恢复和恢复后采样。

3274-token 提示的三轮冷/缓存 TTFT 中位数：Q4_K_M 为 3.33492 秒 / 16.8689 毫秒，
NVFP4 为 3.01954 秒 / 14.6475 毫秒。Q4 的两处缓存/全新 prefill 首次分歧对应
teacher-forced logit deficit 都为 0，满足原有 near-tie 判据，不能说所有冷/缓存序列逐位一致。
NVFP4 本次对应探针未报告这些分歧；测试判据没有放宽。

NVFP4 检查采用 Q4 既有 gate 的临时诊断副本，仅改权重身份断言，链接同一 Engine/Ops。
本次未因纯通信配置变化重跑完整 CUDA 数学套件。上述证据不是通用“智商无损”评分。

### DFlash2 v3：独立容量的实验路线

官方 v3 容器的五层草稿器已接入 text-only TP2。草稿投影保留容器中的 W8 格式、
辅助张量为 BF16；五层全部使用局部注意力，因此移除了不会读取的草稿 Full-KV 池。
主模型的完整历史 KV 没有裁剪。验证配置为 **98304 容量、1024 prefill chunk**；
不承诺两张 16 GB 卡能容纳 180K DFlash，180K 默认仍用 MTP3。

以下为本次交付实现、P2P 开启、INT8 KV、greedy、CUDA Graph 的单次测量，
使用同一份实际 85000-token 的 LRU 代码提示和 draft=7。图预热不计时，不复用前缀。

| 计时 Decode token 数 | Prefill tok/s | 已提交 Decode tok/s | 接受 / 提出草稿 | 接受率 |
|---:|---:|---:|---:|---:|
| 512 | 1185.52 | **99.93** | 422 / 623 | 67.74% |
| 2048，强制窗口 | 1182.63 | **95.87** | 1674 / 2612 | 64.09% |

每次请求还有一个 prefill 输出 token，不计入 decode。512 窗口没有 EOS/EOG；
2048 窗口在第 1150 个总输出 token 出现 `<|im_end|>`，因 benchmark 禁用了模型默认停止而继续生成。
后者是持续执行压力数据，**不是有效长代码完成速度**；两个窗口均不是完整任务评分。
与上一版 argmax-only 路径相比，全部 513 / 2049 个输出 IDs 及接受统计相同。
分片 top-16 的两次单次比较只提高约 0.6–0.8%，不足以证明稳定性能增益。
分片候选与 FP64 lattice oracle、真实 graph/eager 和 64 个 teacher-forcing 位置检查通过；
这些证据不等于通用模型能力无损的评测。

另一份 85K 高代码语料、98304 容量、1024 chunk 的单次强制 2048-token 窗口：
DFlash3/5/7 分别为 79.30/76.10/87.91 tok/s，MTP3 为 86.14。
第 942 个总输出 token 已出现 `<|im_end|>`（DFlash5 为第 917 个），后面仍被强制生成，
所以这些也是压力吞吐，不是有效完成速度。DFlash3/7 与 MTP 的输出 IDs 一致，DFlash5 不一致。
DFlash7 的完整 resident 请求反而更慢：95.40 秒，对比 MTP3 的 92.92 秒。
新测的 1024-token 窗口速度更低；试验性 draft=9/11/15 无收益，已撤回，27B 仍最多七个草稿。

较早的同容量 v3 测量：3072 输入的 DFlash7 为 143.06 tok/s（两轮）；
另一份 85K 代码语料的 DFlash7 为 73.38 tok/s、接受率 47.18%，匹配的 MTP3 为
83.13 tok/s、接受率 78.56%。它们早于本次词表传输优化，不是当前代码重测值。
语料、输出长度和接受率会影响吞吐，不能把 99.93 与另一份语料的 MTP 或 P2P 关闭数据直接算增益。
方法和复现命令见[性能文档](docs/performance.md#dflash2-v3-separate-capacity)。

### DFlash7 / MTP3：非代码、正常停止评测

同一官方 v3 模型，固定 **98304 容量、1024 prefill chunk、TP2、INT8 KV、greedy、
CUDA Graph**，禁用前缀复用，开启模型默认停止；共完成 **80 次请求**。
全部在首个模型结束标记处停止，没有计入 EOS 后续写。
下面 **D/M = DFlash7 / MTP3**，单位为已提交 decode tok/s。
原生短提示的故事/翻译/JSONL/逻辑题分别为 129/395/118/417 个输入 token，各测两次取均值；
其余每个任务/长度/后端只测一次，不能当作稳定均值。

| 实际输入 token | 故事 D/M | 翻译 D/M | JSONL D/M | 逻辑题 D/M |
|---:|---:|---:|---:|---:|
| 原生短提示 | 49.20 / 74.83 | 134.68 / 123.00 | 210.50 / 137.09 | 158.88 / 126.10 |
| 1024 | 51.24 / 76.66 | 127.44 / 117.58 | 206.30 / 135.98 | 157.87 / 125.01 |
| 2048 | 51.09 / 74.41 | 128.45 / 116.14 | 200.78 / 133.63 | 173.32 / 126.34 |
| 4096 | 47.60 / 70.98 | 116.31 / 113.49 | 201.09 / 133.82 | 162.94 / 125.98 |
| 8192 | 49.11 / 70.17 | 125.74 / 116.00 | 198.69 / 132.06 | 166.37 / 125.39 |
| 16384 | 44.80 / 68.16 | 112.08 / 110.15 | 192.60 / 128.33 | 161.27 / 120.91 |
| 32768 | 41.44 / 63.23 | 110.56 / 108.44 | 175.01 / 119.59 | 142.03 / 111.83 |
| 65536 | 36.02 / 55.63 | 85.29 / 89.32 | 148.14 / 104.67 | 122.20 / 98.14 |
| 85000 | 32.26 / 51.67 | 79.61 / 84.86 | 137.65 / 98.77 | 111.05 / 91.60 |

JSONL 在全部长度上均为 DFlash 更快；85K 的接受率 98.30%，decode 比同输出 MTP 快 39.37%。
故事接受率仅约 12–14%，DFlash 反而明显更慢。DFlash prefill 在本轮所有档位都更慢，
因此 decode 优势不代表完整请求更快：85K 逻辑题总耗时为 78.47 / 76.87 秒 D/M。

JSONL 精确字段/算术/顺序检查、逻辑题 CHECK 均通过；翻译只检查了章节和术语表结构。
部分故事违反字数或必含字面项要求，原生 DFlash 故事遗漏 `ORCHID-37`。
36 组任务/长度配对中 30 组输出 IDs 完全一致，85K 四组均一致；
不能由此宣称所有任务逐 token 一致或通用“智商无损”。
[完整 prefill、接受率、输出长度、请求耗时和结果限制](docs/performance.md#stop-aware-non-code-workloads-dflash7-versus-mtp3)
已逐项记录。

## 性能：P2P 关闭

这里区分当前同输入控制组和开启 P2P 前的历史结果。原本的 translated `DMA-FQ` IOMMU
domain 阻止 direct P2P，CUDA 采用经过验证的 UVA D2D staging。当前控制组只在诊断进程
使用 CUDA capability-query shim，未更改系统配置；它不是公开 CLI 的 P2P 开关。

### 当前 NVFP4 v3 同输入控制组

与上方 85K P2P 测量使用相同输入、模型、sampling、KV、MTP、图和输出窗口，各测两轮：

| Prefill tok/s | 已提交 Decode tok/s | Wall decode tok/s | MTP 接受率 |
|---:|---:|---:|---:|
| 1297.94 ± 2.66 | 79.18 ± 0.014 | 79.151 | 79.12% |

开启 P2P 的严格同输入增益为 decode +5.0459%、prefill +0.6578%。

### 早期 NVFP4 数据

以下使用重启前语料，不能替代上面的同输入控制组：

| 模型 / 窗口 | Prefill tok/s | 已提交 Decode tok/s | MTP 接受率 |
|---|---:|---:|---:|
| v2，85K 输入、180K 容量、512 timed 输出 | 1279.44 ± 2.94 | 78.36 ± 0.04 | 79.12% |
| 官方 v3，同一旧 85K 窗口 | 1277.61 ± 3.12 | 78.424 ± 0.0004 | 79.12% |
| 官方 v3，512 输入、2K 容量、512 timed 输出 | 1344.24 ± 33.15 | 98.831 ± 0.027 | — |

官方 v3 还完成了 21-token、100% 接受率的兼容性 smoke，短窗口达到 120.66 tok/s；
这不是稳定长窗口吞吐，也不是 v3 量化本身带来大幅加速的证据。
早期固定 512 输入、256 timed 输出的容量测试如下，每档一轮 warmup 加三轮测量：

| 最大容量 | Wall decode tok/s | Prefill tok/s | MTP 接受率 |
|---:|---:|---:|---:|
| 8192 | 97.65 ± 0.08 | 1369.9 | 70.04% |
| 16384 | 97.67 ± 0.09 | 1370.9 | 70.04% |
| 32768 | 97.76 ± 0.11 | 1370.8 | 70.04% |
| 65536 | 97.55 ± 0.15 | 1369.9 | 70.04% |

### 早期 Q4_K_M 与 LM Studio

Q4_K_M、TP2、INT8 KV、180K 容量、prefill chunk=4096：8192 输入的一次冷 prefill
为 1672.9 tok/s，85000 输入的两轮均值为 1251.44 ± 2.87 tok/s。
85K 输入、MTP3、optimized draft head、CUDA Graph 的两个 128-token decode 窗口测得
**50.68 ± 0.04 tok/s**，MTP 接受率 78.07%。这不是 P2P 开启后的 Q4 decode 测量。

旧容量对照固定 512 输入、256 timed 输出，每档一轮 warmup、三轮测量，greedy、Q8/INT8 KV、
MTP 最大三最小零。NInfer 使用 TP2，LM Studio CUDA 2.33.0 使用自动双卡分配：

| 最大容量 | NInfer Q4_K_M decode tok/s | LM Studio Q4_K_M decode tok/s |
|---:|---:|---:|
| 1024 | 60.06 ± 0.02 | 62.67 ± 0.24 |
| 2048 | 60.12 ± 0.02 | 62.78 ± 0.05 |
| 4096 | 60.11 ± 0.03 | 62.69 ± 0.10 |
| 8192 | 60.13 ± 0.06 | 62.78 ± 0.07 |
| 16384 | 60.14 ± 0.04 | 62.63 ± 0.06 |
| 32768 | 60.16 ± 0.05 | 62.58 ± 0.004 |
| 65536 | 60.06 ± 0.05 | 62.49 ± 0.16 |

同一旧 85K 提示、Q4_K_M、180K 容量的 LM Studio 单次诊断为 35.4977 tok/s，
后端将容量取整到 180224；它不是重复对照。用户先前报告的 45/57 tok/s 缺完整测量元数据，
不当作本轮实测基线，更不能用 Q4_K_M 的 LM Studio 结果证明 NVFP4 的量化质量相同。

### 历史数值限制

当前 DFlash2 v3 数据在上方 P2P 开启区；没有本次 DFlash 的关闭 P2P 配对测量。
旧 DFlash 数据来自不同容器和布局，不能作为当前路线的速度基线。

早期 72.28、75.53、78.90 等 NVFP4 成绩来自修复前的 SwiGLU 中间值路径。
独立 FP64 数学 oracle 检查暴露了 FP8/NVFP4 过早写回 BF16 的误差；修复为 FP32 中间值后，
原判据通过且未放宽容差。旧输出会改变，所以这些成绩不是当前合格数值基线。
更快但改变累加顺序的实验也不是交付路线。详见[数值与历史 profiler 说明](docs/performance.md#p2p-disabled)。

## 构建与模型

需要 64 位 Linux、NVIDIA 驱动、CUDA 12.8、CMake 3.28+、C++20、Ninja、pkg-config、
FFmpeg 开发库及 libcurl。依赖版本不足时可用仓库脚本构建私有依赖：

```bash
tools/v100/build_dependencies.sh
PKG_CONFIG_PATH="$PWD/build/_deps/install/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}" \
cmake -S . -B build-v100 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build build-v100 -j2
```

构建并发按本机情况调整，保留交互余量，整机 CPU 使用率不要超过约 85%。
模型、原始语料和 profiler 文件是本地前提，不随代码仓库提交；C++ 产品只读取 `.ninfer`。
主推 QUASAR，使用现有 Hugging Face CLI 下载固定 v3 成品并校验发布者 checksum，
无需本机转换权重：

```bash
hf download MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer \
  qwen3_8_27b_nvfp4.ninfer SHA256SUMS \
  --revision v3 --local-dir /Models/ninfer-V100X2/quasar-v3
(cd /Models/ninfer-V100X2/quasar-v3 && sha256sum --check SHA256SUMS)
```

其他可选配置：NVFP4 可使用 [Neroued 官方容器](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer)，
本机路径为 `/Models/ninfer-V100X2/qwen3_8_27b_nvfp4.ninfer`。

需要从本地 LM Studio GGUF 转换 Q4_K_M 时，用含 NumPy 的 Python 3.11 环境，例如本机 `.venv`：

```bash
.venv/bin/python3 -m tools.convert.qwen3_8_27b.convert_gguf \
  --model /Models/LM-Studio-models/lmstudio-community/Qwen3.8-27B-GGUF/Qwen3.8-27B-Q4_K_M.gguf \
  --mmproj /Models/LM-Studio-models/lmstudio-community/Qwen3.8-27B-GGUF/mmproj-Qwen3.8-27B-BF16.gguf \
  --out /Models/ninfer-V100X2/qwen3_8_27b_q4_k_m.ninfer
```

GGUF 衍生身份只支持 Text/MTP，保留的 Vision 对象仅用于验证，不代表可以输入图像。

## CLI 和 API 启动

推荐 QUASAR，显式选择模型，180K 容量、2560-token chunk、INT8 KV、optimized-head MTP3：

```bash
NINFER_V100X2_ARTIFACT=/Models/ninfer-V100X2/quasar-v3/qwen3_8_27b_nvfp4.ninfer \
NINFER_V100X2_PREFILL_CHUNK=2560 \
tools/v100/ninfer-v100x2.sh \
  --prompt "用 Python 实现一个有容量上限的任务队列，并解释测试方法。" \
  --max-new 512 --greedy --no-thinking
```

QUASAR、180K、MTP3 API 启动命令：

```bash
env LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  build-v100/apps/ninfer-serve /Models/ninfer-V100X2/quasar-v3/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 8080 \
  --tp 2 --devices 0,1 --max-concurrency 1 \
  --max-context 180000 --kv-capacity 180000 --kv-dtype int8 --prefill-chunk 2560 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --default-max-tokens 65536 --no-thinking
```

DFlash7 将上面两项容量均改为 `98304`，chunk 改为 `1024`；草稿选项换成
`--spec dflash --draft-tokens 7`，去掉 `--lm-head-draft`，匹配 full-head 实测配置。
QUASAR 两种配置均不支持视觉。

Q4_K_M 仍可直接使用不带模型覆盖的默认脚本，默认 chunk 为 4096；官方 NVFP4 则显式指定
`NINFER_V100X2_ARTIFACT=/Models/ninfer-V100X2/qwen3_8_27b_nvfp4.ninfer`，并使用 3072-token
chunk。桌面进程占显存导致 3072 无法通过启动余量检查时，改为 2560，不要削减安全余量。

`--default-max-tokens 65536` 是未指定输出长度时的默认值，仍受剩余上下文容量限制。
服务默认复用兼容前缀；`--no-prefix-reuse` 用于强制冷提示对照。
缓存只复用当前前沿或完整 checkpoint，不是无限 RAM KV 存储。
启动时固定能力与显存预算，请求不能临时开启未加载的执行能力。

OpenAI Chat Completions 请求示例：

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"解释 prefill 与 decode 的区别。"}],"max_tokens":512,"temperature":0}'
```

还提供 `/v1/responses`、`/v1/messages`、`/health` 和 `/v1/models`。
流式输出、tool calls、鉴权及状态语义见[HTTP 文档](docs/serving.md)，具体参数以程序 `--help` 为准。

### 实验性 TP2 视觉

使用官方 NVFP4 成品，不能使用 QUASAR 或 GGUF 衍生 Q4_K_M。在双 16 GB 卡上先用较小的
文本/KV 容量和视觉预算；下面不是最大容量实测配置。`--vision-max-tokens` 限制合并后的视觉
token 数，完整文本加媒体提示仍需满足 `--max-context`。

```bash
env LD_LIBRARY_PATH="$PWD/build/_deps/install/lib:/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  build-v100/apps/ninfer-serve /Models/ninfer-V100X2/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 8080 --tp 2 --devices 0,1 --max-concurrency 1 \
  --max-context 4096 --kv-capacity 4096 --kv-dtype int8 --prefill-chunk 1024 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --vision --vision-max-tokens 1024 --no-prefix-reuse --default-max-tokens 512 --no-thinking
```

这会关闭该服务器所有请求的 Engine 前缀复用，包括文本请求；独立的不可变媒体预处理缓存
不受影响。上面的 QUASAR 文本配置仍保留前缀缓存。
参见[媒体请求示例](docs/serving.md#multimodal-request)与[视觉测试已知限制](tests/README.md#experimental-tp2-vision-status)。

## 致谢与许可

基础来自 [Neroued/ninfer](https://github.com/Neroued/ninfer)、RTX 3060 TP2 路线和
[geoffwatts/ninfer-v100](https://github.com/geoffwatts/ninfer-v100)。Volta NVFP4/TP2 行为参考并
对照过 [plus1998/Ninfer-V100-Duo](https://github.com/plus1998/Ninfer-V100-Duo)；prefill 调研参考
[1CatAI/1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)。DFlash2 实验参考
[Inco AI](https://inco.ai/blog/dflash2/) 及其[草稿模型](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)。
[KVMem](https://github.com/kvmem/kvmem-llama.cpp) 为可选 RAM-KV 实验提供了参考。
原生原型采用 token 匹配检索，不是其 Q/K 向量检索移植；只让部分历史参与注意力是近似模式，
必须显式开启，默认完整上下文注意力不变。

感谢 [Li3age](https://github.com/Li3age) 在
[PR #1](https://github.com/tuxKOH/ninfer-V100X2/pull/1) 贡献 TP2 Vision 和可配置视觉 token 预算。

模型来自 [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)，NVFP4 的混合 FP8/NVFP4
权重来自 [Unsloth](https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4)，容器由
[Neroued](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) 打包。
主推 QUASAR 使用 [QUASAR-QAT 的模型](https://huggingface.co/QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4)，
以及 [MirkoCovizzi 的 NInfer 转换成品](https://huggingface.co/MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer)。

本项目采用 [Apache-2.0](LICENSE)；归属声明见 [NOTICE](NOTICE)，第三方依赖保留各自许可。
