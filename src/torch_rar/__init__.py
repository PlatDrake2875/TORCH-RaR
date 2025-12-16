"""TORCH-RaR: Rubrics as Rewards for Dataset Augmentation.

This package implements the Rubrics as Rewards (RaR) methodology for
augmenting Romanian toxicity detection datasets with evaluation rubrics
and reward signals.

Key components:
- RubricGenerator: Creates instance-specific or predefined evaluation rubrics
- RewardCalculator: Computes explicit/implicit reward aggregation
- AugmentationPipeline: End-to-end dataset augmentation workflow
"""

from torch_rar.config import Settings, load_settings
from torch_rar.data_loader import DatasetLoader
from torch_rar.llm_client import LLMClient
from torch_rar.pipeline import AugmentationPipeline, run_pipeline
from torch_rar.reward_calculator import RewardCalculator
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
    "load_settings",
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
    # Data Loading
    "DatasetLoader",
    # Pipeline
    "AugmentationPipeline",
    "run_pipeline",
]
