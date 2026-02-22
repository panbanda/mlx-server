# mlx-server

[![CI](https://github.com/panbanda/mlx-server/actions/workflows/ci.yml/badge.svg)](https://github.com/panbanda/mlx-server/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/panbanda/mlx-server)](https://github.com/panbanda/mlx-server/releases)
[![Crates.io](https://img.shields.io/crates/v/mlx-server)](https://crates.io/crates/mlx-server)
[![License](https://img.shields.io/badge/license-MIT-blue)](#license)

OpenAI and Anthropic-compatible inference server for Apple Silicon, built in Rust on top of [mlx-rs](https://github.com/oxideai/mlx-rs).

Runs quantized LLMs locally using the Metal GPU with no Python runtime.

## Requirements

- macOS 14+ on Apple Silicon (M1/M2/M3/M4)

## Install

```bash
brew tap panbanda/mlx-server
brew install mlx-server
```

**Build from source** (requires Rust 1.85.0+, Xcode Command Line Tools, and `huggingface-cli`):

```bash
cargo build --release
```

## Usage

```bash
# HuggingFace model ID (resolved from local HF cache)
mlx-server --model mlx-community/Llama-3.1-8B-Instruct-4bit

# Local MLX model directory
mlx-server --model ~/dev/models/My-Custom-Model

# Multiple models
mlx-server --model mlx-community/Llama-3.1-8B-Instruct-4bit --model mlx-community/Qwen3-Coder-Next-4bit
```

The `--model` flag accepts HuggingFace model IDs (`org/name`) or local directory paths. HuggingFace IDs are resolved from the local cache at `~/.cache/huggingface/hub/` (or `$HF_HUB_CACHE` if set, else `$HF_HOME/hub`).

If a model ID is not found in the cache and stdin is a terminal, the server will prompt to download it:

```
Model 'mlx-community/Llama-3.1-8B-Instruct-4bit' not found in HuggingFace cache. Download now? [y/N]
```

In non-interactive mode (piped or scripted), startup fails immediately with a hint to run `huggingface-cli download org/name` manually.

Models must be in **MLX safetensors format**. Pre-quantized weights are available from [mlx-community](https://huggingface.co/mlx-community) on HuggingFace. To convert your own:

```bash
pip install mlx-lm
mlx_lm.convert --hf-path meta-llama/Llama-3.1-8B-Instruct -q --upload-repo your-org/Llama-3.1-8B-4bit-mlx
```

## Configuration

Settings are resolved in order (later wins): defaults, environment variables, CLI flags.

| CLI Flag | Env Variable | Default | Description |
|---|---|---|---|
| `--model` | `MLX_SERVER_MODELS` | *(required)* | Model path or HF model ID (repeatable; env uses JSON array `'["a","b"]'`) |
| `--host` | `MLX_SERVER_HOST` | `0.0.0.0` | Bind address |
| `--port` | `MLX_SERVER_PORT` | `8000` | Bind port |
| `--max-tokens` | `MLX_SERVER_MAX_TOKENS` | `32768` | Default max generation tokens |
| `--api-key` | `MLX_SERVER_API_KEY` | *(none)* | Bearer token for auth (disabled if unset) |
| `--rate-limit` | `MLX_SERVER_RATE_LIMIT` | `0` | Requests per minute per client (0 = disabled) |
| `--timeout` | `MLX_SERVER_TIMEOUT` | `300` | Request timeout in seconds |

Log level is controlled via `RUST_LOG` (e.g., `RUST_LOG=debug`).

## API

### OpenAI-compatible

| Endpoint | Method | Description |
|---|---|---|
| `/v1/chat/completions` | POST | Chat completions (streaming and non-streaming) |
| `/v1/completions` | POST | Text completions |
| `/v1/embeddings` | POST | Token embeddings |
| `/v1/models` | GET | List loaded models |

### Anthropic-compatible

| Endpoint | Method | Description |
|---|---|---|
| `/v1/messages` | POST | Messages API |
| `/v1/messages/count_tokens` | POST | Token counting |

### Health

| Endpoint | Method |
|---|---|
| `/health` | GET |

### Example

```bash
# Chat completion (specify model by name)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Llama-3.1-8B-Instruct-4bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256
  }'

# List all loaded models
curl http://localhost:8000/v1/models
```

## Supported Models

| Architecture | `model_type` | Examples |
|---|---|---|
| LLaMA | `llama` | Llama 3, Llama 3.1, CodeLlama |
| Mistral | `mistral` | Mistral 7B, Mixtral (dense only) |
| Qwen2 | `qwen2` | Qwen2, Qwen2.5 |
| Qwen3 | `qwen3` | Qwen3 |
| Qwen3-Next | `qwen3_next` | Qwen3-Coder (hybrid SSM/attention + MoE) |

## Development

```bash
cargo check
cargo test -- --test-threads=1   # single-threaded to avoid Metal GPU teardown issues
cargo clippy
cargo fmt --check
```

## License

MIT
