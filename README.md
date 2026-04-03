# Qwen3-ASR OpenAI 兼容 Docker 服务

基于官方 `Qwen3-ASR` 运行时封装的本地 ASR 服务，目标是提供最小 OpenAI 兼容接口，便于直接接入已有客户端或 SDK。

## 特性

- NVIDIA GPU-only，推理后端固定为 vLLM。
- OpenAI 兼容面固定为 `GET /v1/models` 和 `POST /v1/audio/transcriptions`。
- 对外暴露两个模型别名：`qwen3-asr-0.6b`、`qwen3-asr-1.7b`。
- 支持 `json`、`text`、`verbose_json`、`srt`、`vtt`。
- 时间戳依赖 `Qwen/Qwen3-ForcedAligner-0.6B`，仅在需要时懒加载。
- 支持空闲卸载，避免单卡常驻多个 ASR 模型。
- 支持可选 Bearer Token 鉴权。

## 目录

```text
.
├── asr_service/            # FastAPI 服务、配置、下载与模型管理
├── scripts/start.sh        # 容器入口
├── tests/                  # 单元测试
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── .github/workflows/
```

## 运行要求

- Linux
- NVIDIA GPU
- Docker Engine + Docker Compose
- 正常可用的 NVIDIA Container Toolkit

## 快速开始

1. 复制环境变量模板：

```bash
cp .env.example .env
```

2. 按需调整 `.env`：

```env
ASR_ENABLED_MODELS=0.6b,1.7b
ASR_DOWNLOAD_SOURCE=modelscope
ASR_ENABLE_ALIGNER=true
ASR_API_KEY=
ASR_PORT=8000
```

3. 启动服务：

```bash
docker compose up --build
```

4. 检查健康状态：

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/readyz
```

5. 查看可用模型：

```bash
curl http://127.0.0.1:8000/v1/models
```

如果配置了 `ASR_API_KEY`，需要带上：

```bash
-H "Authorization: Bearer <your-token>"
```

## 转写请求示例

### `json`

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@sample.wav" \
  -F "model=qwen3-asr-0.6b" \
  -F "response_format=json"
```

### `verbose_json` + 词级/段级时间戳

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@sample.wav" \
  -F "model=qwen3-asr-1.7b" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities[]=word" \
  -F "timestamp_granularities[]=segment" \
  -F "temperature=0"
```

### `srt`

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -F "file=@sample.wav" \
  -F "model=qwen3-asr-0.6b" \
  -F "response_format=srt"
```

## 接口行为

### `GET /v1/models`

- 只返回当前启用的 ASR 模型。
- aligner 不会暴露为可调用模型。

### `POST /v1/audio/transcriptions`

支持字段：

- `file`
- `model`
- `language`
- `prompt`
- `response_format`
- `timestamp_granularities[]`
- `temperature`

限制：

- `model` 仅接受 `qwen3-asr-0.6b` / `qwen3-asr-1.7b`
- `temperature` 仅接受空值或 `0`
- aligner 关闭时，请求时间戳、`srt`、`vtt` 会返回 `400`

## 关键环境变量

| 变量 | 默认值 | 说明 |
|---|---:|---|
| `ASR_ENABLED_MODELS` | `0.6b,1.7b` | 启用的模型列表 |
| `ASR_DOWNLOAD_SOURCE` | `modelscope` | `modelscope` 或 `huggingface` |
| `ASR_ENABLE_ALIGNER` | `true` | 是否允许时间戳能力 |
| `ASR_IDLE_UNLOAD_SECONDS` | `900` | 空闲多少秒后卸载显存中的模型 |
| `ASR_GPU_MEMORY_UTILIZATION` | `0.8` | 传给 vLLM 的显存占用上限 |
| `ASR_API_KEY` | 空 | 非空时启用 Bearer Token 校验 |
| `ASR_MODELS_DIR` | `/models` | 容器内模型目录 |
| `ASR_PORT` | `8000` | 服务端口 |

## GitHub Actions 手动发布镜像

仓库内提供了手动触发的工作流：

- 路径：`.github/workflows/publish-image.yml`
- 触发方式：GitHub 页面 `Actions` -> `Publish Docker Image` -> `Run workflow`
- 推送目标：`ghcr.io/<owner>/<repo>`

工作流输入：

- `tag`：可选，自定义额外标签
- `push_latest`：是否额外推送 `latest`
- `bundle_flash_attention`：构建时是否安装 FlashAttention

默认会至少推送两个标签：

- `ghcr.io/<owner>/<repo>:sha-<short-sha>`
- `ghcr.io/<owner>/<repo>:<branch-name>`

需要的仓库权限：

- Workflow 具备 `packages: write`
- 允许 `GITHUB_TOKEN` 写入 GHCR 包

如果是组织仓库，并且组织策略限制了包发布，需要在组织或仓库设置里放开 GHCR 发布权限。

## 本地测试

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## 参考

- 官方 Qwen3-ASR: https://github.com/QwenLM/Qwen3-ASR
- Forced Aligner: https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B
- vLLM 清理接口: https://github.com/vllm-project/vllm/blob/main/vllm/distributed/parallel_state.py
