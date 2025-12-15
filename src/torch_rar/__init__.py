"""TORCH-RaR: Rubrics as Rewards for Dataset Augmentation."""

from torch_rar.config import Settings, load_settings
from torch_rar.data_loader import DatasetLoader
from torch_rar.llm_client import LLMClient
from torch_rar.pipeline import AugmentationPipeline, run_pipeline
from torch_rar.reward_calculator import RewardCalculator
from torch_rar.rubric_generator import RubricGenerator

__version__ = "0.1.0"
__all__ = [
    "Settings",
    "load_settings",
    "LLMClient",
    "RubricGenerator",
    "RewardCalculator",
    "DatasetLoader",
    "AugmentationPipeline",
    "run_pipeline",
]
