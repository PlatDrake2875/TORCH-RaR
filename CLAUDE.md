# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TORCH-RaR - A dataset augmentation tool implementing the **Rubrics as Rewards (RaR)** method from the paper "Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains". The project augments toxicity detection datasets by generating instance-specific evaluation rubrics and computing reward signals.

Key features:
- Instance-specific rubric generation for toxicity evaluation
- Both explicit and implicit reward aggregation strategies
- Support for OpenRouter, vLLM (Docker), and LiteLLM proxy for LLM inference
- HuggingFace dataset integration
- YAML-based configuration with environment variable substitution

## Build and Development Commands

```bash
# Install dependencies with uv
uv sync

# Configure settings (edit settings.yaml)
# Set your API key: openrouter_api_key: your_key_here
# Or use env var: openrouter_api_key: ${OPENROUTER_API_KEY}

# Run the pipeline
uv run python main.py run --limit 10

# Test configuration
uv run python main.py test

# Run with predefined rubrics (faster)
uv run python main.py run --limit 100 --predefined-rubrics

# Use custom config file
uv run python main.py --config custom_settings.yaml run --limit 10

# Start vLLM server via Docker
docker-compose up vllm

# Start with LiteLLM proxy
docker-compose --profile with-proxy up
```

## Architecture

```
src/torch_rar/
├── __init__.py           # Package exports
├── config.py             # Settings loaded from settings.yaml
├── llm_client.py         # LiteLLM wrapper for OpenRouter/vLLM
├── data_loader.py        # HuggingFace dataset loading
├── rubric_generator.py   # Instance-specific rubric generation
├── reward_calculator.py  # Explicit/implicit reward aggregation
└── pipeline.py           # Main augmentation pipeline

main.py                   # CLI entry point
settings.yaml             # Main configuration file
docker-compose.yml        # vLLM and LiteLLM proxy containers
litellm_config.yaml       # LiteLLM proxy configuration
```

## Configuration

Settings are loaded from `settings.yaml`. The file supports environment variable substitution using `${VAR_NAME}` syntax.

Key settings:
- `llm_provider`: openrouter | vllm | litellm_proxy
- `openrouter_api_key`: Your API key (or `${OPENROUTER_API_KEY}`)
- `rubric_generator_model`: Model for generating rubrics
- `judge_model`: Model for evaluating responses
- `dataset_name`: HuggingFace dataset to augment

## Key Concepts (from RaR paper)

1. **Rubrics**: Instance-specific evaluation criteria with categories:
   - Essential: Critical factors (weight 1.0)
   - Important: Significant factors (weight 0.7)
   - Optional: Nice-to-have (weight 0.3)
   - Pitfall: Common mistakes to avoid (weight 0.9)

2. **Explicit Aggregation**: Each criterion evaluated independently, normalized weighted sum
3. **Implicit Aggregation**: All rubrics passed to LLM judge for holistic scoring (1-10 Likert)

## Deployment Scripts

The `scripts/deploy.sh` script provides automated deployment and management:

```bash
# Interactive deployment menu
./scripts/deploy.sh

# Full setup with provider selection
./scripts/deploy.sh setup [vllm|openrouter|litellm_proxy]

# Service management
./scripts/deploy.sh start [vllm|litellm]   # Start services
./scripts/deploy.sh stop                    # Stop all services
./scripts/deploy.sh status                  # Show service status
./scripts/deploy.sh logs [vllm|litellm]    # View service logs

# Testing and validation
./scripts/deploy.sh test      # Test LLM connection
./scripts/deploy.sh validate  # Validate config files
```
