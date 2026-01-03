"""TORCH-RaR: Rubrics as Rewards for Dataset Augmentation.

This package implements the Rubrics as Rewards (RaR) methodology for
augmenting Romanian toxicity detection datasets with evaluation rubrics
and reward signals.

Key components:
- RubricGenerator: Creates instance-specific or predefined evaluation rubrics
- RewardCalculator: Computes explicit/implicit reward aggregation
- AugmentationPipeline: End-to-end dataset augmentation workflow
"""

from torch_rar.config import RubricWeights, Settings, get_settings, load_settings, reset_settings
from torch_rar.data_loader import AugmentedSample, DatasetLoader, ToxicitySample
from torch_rar.exceptions import (
    ConfigurationError,
    DatasetError,
    JSONParseError,
    LLMClientError,
    RewardCalculationError,
    RubricGenerationError,
    TorchRarError,
    ValidationError,
)
from torch_rar.json_utils import (
    extract_boolean_from_response,
    extract_json_from_response,
    extract_rating_from_response,
)
from torch_rar.llm_client import LLMClient
from torch_rar.pipeline import AugmentationPipeline, PipelineStats, run_pipeline
from torch_rar.reward_calculator import RewardCalculator, RewardResult, RubricEvaluation
from torch_rar.rubric_generator import (
    RubricCategory,
    RubricGenerator,
    RubricItem,
    get_rubric_by_id,
    get_rubrics_by_category,
    get_torch_rar_rubrics,
)

__version__ = "0.1.0"
__all__ = [
    # Configuration
    "Settings",
    "RubricWeights",
    "load_settings",
    "get_settings",
    "reset_settings",
    # Exceptions
    "TorchRarError",
    "ConfigurationError",
    "LLMClientError",
    "JSONParseError",
    "RubricGenerationError",
    "RewardCalculationError",
    "DatasetError",
    "ValidationError",
    # JSON Utilities
    "extract_json_from_response",
    "extract_boolean_from_response",
    "extract_rating_from_response",
    # LLM Client
    "LLMClient",
    # Rubric Generation
    "RubricGenerator",
    "RubricItem",
    "RubricCategory",
    "get_torch_rar_rubrics",
    "get_rubric_by_id",
    "get_rubrics_by_category",
    # Reward Calculation
    "RewardCalculator",
    "RewardResult",
    "RubricEvaluation",
    # Data Loading
    "DatasetLoader",
    "ToxicitySample",
    "AugmentedSample",
    # Pipeline
    "AugmentationPipeline",
    "PipelineStats",
    "run_pipeline",
]
