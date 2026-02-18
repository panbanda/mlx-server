# mlx-server

OpenAI and Anthropic-compatible inference server for Apple Silicon, built in Rust on top of [mlx-rs](https://github.com/oxideai/mlx-rs).

Runs quantized LLMs locally using the Metal GPU with no Python runtime.

## Supported Models

| Architecture | `model_type` | Examples |
|---|---|---|
| LLaMA | `llama` | Llama 3, Llama 3.1, CodeLlama |
| Mistral | `mistral` | Mistral 7B, Mixtral (dense only) |
| Qwen2 | `qwen2` | Qwen2, Qwen2.5 |
| Qwen3 | `qwen3` | Qwen3 |
| Qwen3-Next | `qwen3_next` | Qwen3-Coder (hybrid SSM/attention + MoE) |

Models must be in **MLX safetensors format**. Pre-quantized weights are available from the [mlx-community](https://huggingface.co/mlx-community) org on HuggingFace. To convert your own:

```bash
pip install mlx-lm
mlx_lm.convert --hf-path meta-llama/Llama-3.1-8B-Instruct -q --upload-repo your-org/Llama-3.1-8B-4bit-mlx
```

## Prerequisites

- macOS 14+ on Apple Silicon (M1/M2/M3/M4)
- Rust 1.85.0+
- Xcode Command Line Tools (for Metal compiler)

## Build

```bash
cargo build --release
```

## Usage

```bash
# Point at a local MLX model directory
mlx-server --model ~/.cache/huggingface/hub/models--mlx-community--Llama-3.1-8B-Instruct-4bit

# Or use a HuggingFace model ID (downloads on first use)
mlx-server --model mlx-community/Llama-3.1-8B-Instruct-4bit
```

### Configuration

Settings are resolved in order (later wins): defaults, environment variables, CLI flags.

| CLI Flag | Env Variable | Default | Description |
|---|---|---|---|
| `--model` | `MLX_SERVER_MODEL` | `~/.cache/huggingface/hub` | Model directory or HF model ID |
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

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Server health check |

### Example

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.1-8b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256
  }'
```

With authentication:

```bash
mlx-server --api-key YOUR_API_KEY

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{"model": "llama", "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Project Structure

```text
crates/
  mlx-models/    Model architectures (transformer, qwen3_next) and weight loading
  mlx-engine/    Inference engine (tokenization, generation loop, prompt caching)
  mlx-server/    HTTP server (Axum routes, OpenAI/Anthropic adapters, config)
```

## Development

```bash
cargo check                              # Type check
cargo test -- --test-threads=1           # Run tests (single-threaded for Metal)
cargo clippy                             # Lint
cargo fmt --check                        # Format check
```

Tests must run single-threaded (`--test-threads=1`) to avoid Metal GPU teardown issues.

## License

MIT OR Apache-2.0
