# TORCH-RaR

**Rubrics as Rewards for Toxicity Dataset Augmentation**

A dataset augmentation tool implementing the **Rubrics as Rewards (RaR)** method from the paper "Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains". This tool augments toxicity detection datasets by generating instance-specific evaluation rubrics and computing reward signals.

## Features

- **Instance-specific rubric generation** for toxicity evaluation
- **Multiple reward aggregation strategies**: Explicit (weighted sum) and Implicit (LLM judge)
- **Flexible LLM provider support**: OpenRouter (cloud), vLLM (local GPU), LiteLLM proxy
- **HuggingFace dataset integration** for seamless data loading
- **Async processing** with configurable concurrency

## Quick Start

### Using the Deployment Script (Recommended)

```bash
# Interactive setup
./scripts/deploy.sh

# Or non-interactive setup with a specific provider
./scripts/deploy.sh setup openrouter   # Cloud API (easiest)
./scripts/deploy.sh setup vllm         # Local GPU inference
./scripts/deploy.sh setup litellm_proxy # Docker proxy
```

### Manual Setup

```bash
# 1. Install uv package manager (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 4. Run the pipeline
uv run python main.py run --limit 10
```

## LLM Providers

### OpenRouter (Cloud API)

Best for quick testing without GPU requirements.

```bash
# Set your API key
export OPENROUTER_API_KEY=your_key_here

# Or add to .env file
echo "OPENROUTER_API_KEY=your_key_here" >> .env

# Ensure settings.yaml has:
# llm_provider: openrouter
```

### vLLM (Local GPU)

Best for production with NVIDIA GPU.

```bash
# Start vLLM container
docker-compose up vllm

# Or use deploy script
./scripts/deploy.sh start vllm

# Ensure settings.yaml has:
# llm_provider: vllm
```

#### Using GGUF Models (Recommended for Limited VRAM)

GGUF quantized models allow running larger models on GPUs with limited VRAM (e.g., 8GB).

```bash
# 1. Download a GGUF model (Q3_K_M fits ~4GB, good for 8GB VRAM)
mkdir -p ~/.cache/gguf
wget -O ~/.cache/gguf/qwen2.5-7b-instruct-q3_k_m.gguf \
  "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q3_k_m.gguf"

# 2. Run vLLM with GGUF model
docker run -d --name torch-rar-vllm-gguf --gpus all -p 8000:8000 \
  -v ~/.cache/gguf:/models \
  vllm/vllm-openai:latest \
  --model /models/qwen2.5-7b-instruct-q3_k_m.gguf \
  --tokenizer Qwen/Qwen2.5-7B-Instruct \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --host 0.0.0.0 --port 8000

# 3. Update settings.yaml:
# vllm_model_name: /models/qwen2.5-7b-instruct-q3_k_m.gguf
```

**GGUF Quantization Options:**
| Quantization | Size | Quality | VRAM Required |
|--------------|------|---------|---------------|
| Q2_K | ~3GB | Lower | ~4GB |
| Q3_K_M | ~3.8GB | Good | ~5GB |
| Q4_K_M | ~4.5GB | Better | ~6GB |
| Q5_K_M | ~5GB | High | ~7GB |

**Note:** vLLM requires single-file GGUF models. Use `gguf-split` to merge multi-part files.

### LiteLLM Proxy

Routes requests to multiple backends with unified API.

```bash
# Start both vLLM and LiteLLM proxy
docker-compose --profile with-proxy up

# Or use deploy script
./scripts/deploy.sh start litellm

# Ensure settings.yaml has:
# llm_provider: litellm_proxy
```

## Usage

```bash
# Run with default settings (10 samples, both reward methods)
uv run python main.py run --limit 10

# Use predefined rubrics for faster processing
uv run python main.py run --limit 100 --predefined-rubrics

# Use only implicit reward calculation
uv run python main.py run --limit 50 --reward-method implicit

# Custom column names for your dataset
uv run python main.py run --limit 20 --text-column comment --label-column toxic

# Output as JSON instead of Parquet
uv run python main.py run --limit 10 --output-format json

# Test configuration and API connectivity
uv run python main.py test -v
```

## Deployment Script Commands

```bash
./scripts/deploy.sh                    # Interactive menu
./scripts/deploy.sh setup [provider]   # Full setup (vllm|openrouter|litellm_proxy)
./scripts/deploy.sh start [service]    # Start services (vllm|litellm)
./scripts/deploy.sh stop               # Stop all services
./scripts/deploy.sh status             # Show service status
./scripts/deploy.sh logs [service]     # View service logs
./scripts/deploy.sh test               # Test LLM connection
./scripts/deploy.sh validate           # Validate config files
./scripts/deploy.sh help               # Show help
```

## Configuration

Edit `settings.yaml` to customize:

```yaml
# LLM provider selection
llm_provider: openrouter  # openrouter | vllm | litellm_proxy

# Models
rubric_generator_model: openrouter/openai/gpt-4o
judge_model: openrouter/openai/gpt-4o-mini

# Dataset
dataset_name: olimpia20/toxicity-dataset-ro-master
dataset_split: train

# Processing
batch_size: 10
max_concurrent_requests: 5

# Rubric settings
min_rubric_items: 7
max_rubric_items: 20

# Reward weights (from RaR paper)
weight_essential: 1.0
weight_important: 0.7
weight_optional: 0.3
weight_pitfall: 0.9
```

## Project Structure

```
TORCH/
├── src/torch_rar/         # Main Python package
│   ├── config.py          # Settings and configuration
│   ├── llm_client.py      # LiteLLM wrapper for API calls
│   ├── data_loader.py     # HuggingFace dataset loading
│   ├── rubric_generator.py # Instance-specific rubric generation
│   ├── reward_calculator.py # Reward computation (explicit/implicit)
│   └── pipeline.py        # Main augmentation pipeline
├── scripts/
│   └── deploy.sh          # Deployment automation script
├── docs/                  # Documentation and research papers
├── main.py                # CLI entry point
├── docker-compose.yml     # Docker container configuration
├── litellm_config.yaml    # LiteLLM proxy model routing
├── settings.yaml          # Application configuration
└── .env.example           # Environment variables template
```

## Key Concepts (from RaR Paper)

### Rubric Categories

- **Essential**: Critical factors that must be evaluated (weight: 1.0)
- **Important**: Significant factors for quality assessment (weight: 0.7)
- **Optional**: Nice-to-have criteria (weight: 0.3)
- **Pitfall**: Common mistakes to penalize (weight: 0.9)

### Aggregation Methods

1. **Explicit Aggregation**: Each rubric criterion is evaluated independently by the LLM, scores are combined using normalized weighted sum
2. **Implicit Aggregation**: All rubrics are passed to the LLM judge for holistic scoring (1-10 Likert scale)

## Requirements

- Python 3.10+
- Docker & Docker Compose (for vLLM/LiteLLM providers)
- NVIDIA GPU with CUDA (for vLLM local inference)
- OpenRouter API key (for cloud inference)

## Documentation

See the `docs/` folder for:
- Research paper: Rubrics as Rewards methodology
- Project milestone documentation

## License

MIT License
