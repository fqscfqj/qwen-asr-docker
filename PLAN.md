# Qwen3-ASR OpenAI 兼容 Docker ASR 服务

**摘要**
- 基于官方 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 开源运行时实现本地服务，`Qwen3-ASR-Toolkit` 只作为接口形态和长音频处理思路参考，因为 toolkit 本身是 DashScope API 客户端，不是本地模型服务。
- 服务范围固定为 NVIDIA GPU-only、vLLM 后端、OpenAI 最小兼容面：`GET /v1/models` + `POST /v1/audio/transcriptions`。
- 对外暴露两个模型别名：`qwen3-asr-0.6b` 和 `qwen3-asr-1.7b`。如果 compose 同时启用两者，启动阶段都预下载，但显存里同一时刻只保留一个活动 ASR 模型。
- 时间戳依赖 `Qwen/Qwen3-ForcedAligner-0.6B`。启用后只在请求需要 timestamps 时懒加载，对齐模型也参与空闲卸载。

**Public Interfaces**
- 提供 `docker-compose.yml` 和 `.env.example`，核心变量固定为：`ASR_ENABLED_MODELS=0.6b,1.7b`、`ASR_DOWNLOAD_SOURCE=modelscope|huggingface`、`ASR_ENABLE_ALIGNER=true|false`、`ASR_IDLE_UNLOAD_SECONDS`、`ASR_GPU_MEMORY_UTILIZATION`、`ASR_API_KEY`、`ASR_MODELS_DIR=/models`、`ASR_PORT=8000`。
- `GET /v1/models` 仅返回已启用 ASR 模型，使用 OpenAI 风格 `object=list` / `data[]`；不把 aligner 暴露成可调用 model。
- `POST /v1/audio/transcriptions` 接收 OpenAI 风格 multipart 字段：`file`、`model`、`language`、`prompt`、`response_format`、`timestamp_granularities[]`、`temperature`。
- 对外 `model` 只接受 `qwen3-asr-0.6b` / `qwen3-asr-1.7b`，服务内部映射到本地目录和上游 repo id。
- `response_format` 支持 `json`、`text`、`verbose_json`、`srt`、`vtt`；`timestamp_granularities` 支持 `word`、`segment`。
- `temperature` 只接受空值或 `0`；其他值返回 `400`，因为底层是固定 ASR 解码而不是采样解码。
- 当 aligner 未启用且请求了 `timestamp_granularities`、`srt`、`vtt`，或需要带 timestamps 的 `verbose_json` 时，返回 `400` 明确报错。

**Implementation Changes**
- 使用一份自定义 CUDA 镜像，依赖栈对齐官方 [`docker/Dockerfile-qwen3-asr-cu128`](https://github.com/QwenLM/Qwen3-ASR/blob/main/docker/Dockerfile-qwen3-asr-cu128)，额外安装 FastAPI/Uvicorn、`python-multipart`、`modelscope`、`huggingface_hub[cli]`、`ffmpeg`。
- 启动脚本先做配置校验，再按 `ASR_DOWNLOAD_SOURCE` 下载已启用模型到持久卷；目录已存在且具备权重文件时跳过重复下载；若 `ASR_ENABLE_ALIGNER=true`，同时预下载 `Qwen/Qwen3-ForcedAligner-0.6B`。
- 服务层通过 `Qwen3ASRModel.LLM(...)` 从本地模型目录启动 vLLM；对齐模型通过 `Qwen3ForcedAligner.from_pretrained(...)` 从本地目录加载。
- 设计单例 `ModelManager`，统一维护“已下载模型元数据、当前活动模型、aligner 状态、最后使用时间、进行中请求计数”。
- 懒加载规则固定：容器启动只下载不占显存；首个请求按 `model` 加载对应 vLLM 模型；只有请求了 timestamps 或 `srt/vtt` 时才加载 aligner。
- 多模型规则固定：如果当前活动模型与请求模型不同，先阻塞新请求，卸载旧 vLLM 实例，再加载新实例；单卡上不尝试双模型常驻。
- 卸载规则固定：空闲超过 `ASR_IDLE_UNLOAD_SECONDS` 且无进行中请求时，删除模型对象、执行 `gc.collect()`、`torch.cuda.empty_cache()`，并调用 vLLM 的 [`cleanup_dist_env_and_memory`](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/distributed/parallel_state.py) 清理分布式和显存状态。
- 转写映射规则固定：`prompt` 映射为 Qwen3-ASR 的 `context`，`language` 透传；`json/text` 返回纯文本；`verbose_json` 返回 `text`、`language`、`duration` 和按需生成的 `words` / `segments`；`srt/vtt` 基于 timestamps 生成字幕文本。
- `words` 直接来自 forced aligner token spans；`segments` 采用固定切分规则，避免实现时再做选择：句末标点断段，或相邻词间隔 `>0.8s` 断段，或单段时长 `>12s` 断段，或单段字符数 `>120` 断段。
- 额外提供 `GET /healthz` 和 `GET /readyz` 供 compose/监控使用；`readyz` 只有在启动下载完成且服务可接请求时才返回成功。
- 鉴权做成可开关的共享 Bearer Token：`ASR_API_KEY` 非空时校验 `Authorization: Bearer ...`，为空时关闭鉴权。

**Test Plan**
- 配置解析测试：模型别名、下载源、aligner 开关、TTL、API key 行为都正确生效。
- 下载测试：`modelscope` 和 `huggingface` 两套下载命令生成正确，已有缓存时不重复下载。
- API 测试：`/v1/models` 只列出启用模型；`/v1/audio/transcriptions` 在 `json`、`text`、`verbose_json`、`srt`、`vtt` 下返回正确格式。
- 时间戳测试：aligner 开启时 `word`/`segment` 都能返回；aligner 关闭时相关请求稳定返回 `400`。
- 生命周期测试：首个请求触发懒加载；`0.6b -> 1.7b` 请求切换会先卸载旧模型；空闲超时后显存清理生效；超时后下一次请求能自动重新加载。
- 鉴权测试：`ASR_API_KEY` 开启后，无 token 和错误 token 返回 `401`，正确 token 正常访问。
- Docker 集成测试：`docker compose up` 后自动下载选中模型，`curl` 或 OpenAI SDK 调用 `/v1/models` 和 `/v1/audio/transcriptions` 可用。

**Assumptions**
- 只支持 NVIDIA GPU 容器，不提供 CPU 路径。
- OpenAI 兼容范围固定为 `/v1/models` 和 `/v1/audio/transcriptions`，不实现 `/v1/chat/completions`。
- 预下载是同步启动行为，下载失败即容器启动失败，不做后台补下载。
- `Qwen3-ASR-Toolkit` 仅作参考；实际推理基线采用官方 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)，因为 toolkit 官方 README 说明它本质上是 DashScope API 工具。
- 关键参考：[`Qwen3-ASR-Toolkit` README](https://github.com/QwenLM/Qwen3-ASR-Toolkit)、[`Qwen3-ASR` README](https://github.com/QwenLM/Qwen3-ASR)、[`Qwen/Qwen3-ForcedAligner-0.6B` model card](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B)。
