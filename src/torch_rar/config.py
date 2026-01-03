"""Configuration settings for TORCH-RaR loaded from settings.yaml."""

import os
import re
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class RubricWeights:
    """Standard weights for TORCH-RaR rubric categories.

    These weights follow the RaR paper Section 5.1 methodology for
    Romanian toxicity detection. Weights are used in explicit aggregation:

        r(x, ŷ) = Σⱼ(wⱼ · cⱼ(x, ŷ)) / Σⱼ|wⱼ|

    Essential criteria have highest weights (critical indicators).
    Important criteria have moderate weights (contextual factors).
    Pitfall criteria have negative weights (penalties for errors).
    """

    # Essential criteria weights (E1-E4)
    E1_CORRECT_LABEL = 1.0
    E2_PERSONAL_ATTACK = 0.95
    E3_THREAT_DETECTION = 0.90
    E4_GROUP_HATRED = 0.90

    # Important criteria weights (I1-I4)
    I1_CONTEXTUAL = 0.70
    I2_EMOTIONAL = 0.65
    I3_SARCASM = 0.60
    I4_POLITICAL = 0.60

    # Pitfall criteria weights (P1-P3) - negative for penalties
    P1_FALSE_POSITIVE = -0.60
    P2_FALSE_NEGATIVE = -0.65
    P3_CONTEXT_FREE = -0.50

    # Default category weights for prompt-based rubrics
    ESSENTIAL_DEFAULT = 1.0
    IMPORTANT_DEFAULT = 0.7
    OPTIONAL_DEFAULT = 0.3
    PITFALL_DEFAULT = -0.9


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENROUTER = "openrouter"
    VLLM = "vllm"
    LITELLM_PROXY = "litellm_proxy"


def _expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in values.

    Supports ${VAR_NAME} syntax for environment variable substitution.
    """
    if isinstance(value, str):
        # Match ${VAR_NAME} pattern
        pattern = r"\$\{([^}]+)\}"
        matches = re.findall(pattern, value)
        for var_name in matches:
            env_value = os.environ.get(var_name, "")
            value = value.replace(f"${{{var_name}}}", env_value)
        return value if value else None
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    return value


def _find_settings_file() -> Path:
    """Find the settings.yaml file.

    Searches in order:
    1. Current working directory
    2. Project root (where pyproject.toml is)
    3. Package directory
    """
    # Current directory
    cwd_path = Path.cwd() / "settings.yaml"
    if cwd_path.exists():
        return cwd_path

    # Look for project root by finding pyproject.toml
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            settings_path = parent / "settings.yaml"
            if settings_path.exists():
                return settings_path

    # Fallback to package directory
    package_dir = Path(__file__).parent.parent.parent
    settings_path = package_dir / "settings.yaml"
    if settings_path.exists():
        return settings_path

    raise FileNotFoundError(
        "settings.yaml not found. Please create one from settings.yaml.example"
    )


class Settings(BaseModel):
    """Application settings loaded from settings.yaml."""

    # LLM Provider Settings
    llm_provider: LLMProvider = Field(
        default=LLMProvider.OPENROUTER,
        description="LLM provider to use (openrouter, vllm, litellm_proxy)",
    )

    # OpenRouter Settings
    openrouter_api_key: Optional[str] = Field(
        default=None,
        description="OpenRouter API key",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL",
    )

    # vLLM Settings (for local Docker deployment)
    vllm_base_url: str = Field(
        default="http://localhost:8000/v1",
        description="vLLM server base URL",
    )
    vllm_model_name: str = Field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        description="Model name for vLLM server",
    )

    # LiteLLM Proxy Settings
    litellm_proxy_url: str = Field(
        default="http://localhost:4000",
        description="LiteLLM proxy server URL",
    )
    litellm_api_key: Optional[str] = Field(
        default=None,
        description="LiteLLM proxy API key (if configured)",
    )

    # Model Selection
    rubric_generator_model: str = Field(
        default="openrouter/openai/gpt-4o",
        description="Model for generating rubrics",
    )
    judge_model: str = Field(
        default="openrouter/openai/gpt-4o-mini",
        description="Model for evaluating responses (LLM-as-Judge)",
    )

    # Dataset Settings
    dataset_name: str = Field(
        default="olimpia20/toxicity-dataset-ro-master",
        description="HuggingFace dataset to augment",
    )
    dataset_split: str = Field(
        default="train",
        description="Dataset split to use",
    )
    output_dir: str = Field(
        default="./output",
        description="Directory for output files",
    )

    # Rubric Settings
    min_rubric_items: int = Field(default=7, description="Minimum rubric items per prompt")
    max_rubric_items: int = Field(default=20, description="Maximum rubric items per prompt")

    # Processing Settings
    batch_size: int = Field(default=10, description="Batch size for processing")
    max_concurrent_requests: int = Field(
        default=5, description="Max concurrent LLM API requests"
    )
    request_timeout: int = Field(default=120, description="API request timeout in seconds")
    max_retries: int = Field(default=3, description="Max retries for failed requests")

    # Reward Weights (for explicit aggregation)
    weight_essential: float = Field(default=1.0, description="Weight for Essential criteria")
    weight_important: float = Field(default=0.7, description="Weight for Important criteria")
    weight_optional: float = Field(default=0.3, description="Weight for Optional criteria")
    weight_pitfall: float = Field(default=0.9, description="Weight for Pitfall criteria")

    model_config = {"extra": "ignore"}

    def get_litellm_model_name(self, model_type: str = "judge") -> str:
        """Get the appropriate model name for LiteLLM based on provider."""
        if model_type == "rubric":
            model = self.rubric_generator_model
        else:
            model = self.judge_model

        if self.llm_provider == LLMProvider.OPENROUTER:
            # OpenRouter models are prefixed with "openrouter/"
            if not model.startswith("openrouter/"):
                return f"openrouter/{model}"
            return model
        elif self.llm_provider == LLMProvider.VLLM:
            # vLLM uses the model name directly
            return f"hosted_vllm/{self.vllm_model_name}"
        else:
            # LiteLLM proxy - use model name as-is
            return model

    def get_api_base(self) -> Optional[str]:
        """Get the API base URL based on provider."""
        if self.llm_provider == LLMProvider.OPENROUTER:
            return self.openrouter_base_url
        elif self.llm_provider == LLMProvider.VLLM:
            return self.vllm_base_url
        elif self.llm_provider == LLMProvider.LITELLM_PROXY:
            return self.litellm_proxy_url
        return None

    def get_api_key(self) -> Optional[str]:
        """Get the API key based on provider."""
        if self.llm_provider == LLMProvider.OPENROUTER:
            return self.openrouter_api_key
        elif self.llm_provider == LLMProvider.LITELLM_PROXY:
            return self.litellm_api_key
        return None


def load_settings(config_path: Optional[str | Path] = None) -> Settings:
    """Load settings from YAML file.

    Args:
        config_path: Path to settings.yaml. If None, searches default locations.

    Returns:
        Settings object with loaded configuration.
    """
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Settings file not found: {path}")
    else:
        path = _find_settings_file()

    with open(path) as f:
        raw_config = yaml.safe_load(f) or {}

    # Expand environment variables
    config = _expand_env_vars(raw_config)

    return Settings(**config)


# Global settings instance (lazy loaded)
_settings: Optional[Settings] = None


def get_settings(
    config_path: Optional[str | Path] = None,
    force_reload: bool = False,
) -> Settings:
    """Get the global settings instance.

    Args:
        config_path: Optional path to settings.yaml for first load.
        force_reload: If True, reload settings even if already loaded.

    Returns:
        Settings instance.
    """
    global _settings
    if _settings is None or force_reload:
        _settings = load_settings(config_path)
    return _settings


def reset_settings() -> None:
    """Reset the global settings instance.

    Useful for testing to ensure clean state between tests.
    After calling this, the next call to get_settings() will
    reload settings from the configuration file.
    """
    global _settings
    _settings = None
