"""Dataset loading and management for toxicity data augmentation."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from torch_rar.config import Settings
from torch_rar.exceptions import DatasetError

logger = logging.getLogger(__name__)


@dataclass
class ToxicitySample:
    """A single sample from the toxicity dataset."""

    id: str
    text: str
    label: Optional[int] = None  # 0 = non-toxic, 1 = toxic
    metadata: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "label": self.label,
            "metadata": self.metadata or {},
        }


@dataclass
class AugmentedSample:
    """An augmented sample with rubrics and rewards."""

    original: ToxicitySample
    rubrics: list[dict[str, Any]]
    reward_explicit: Optional[float] = None
    reward_implicit: Optional[float] = None
    generated_response: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for saving."""
        return {
            "id": self.original.id,
            "text": self.original.text,
            "label": self.original.label,
            "rubrics": self.rubrics,
            "reward_explicit": self.reward_explicit,
            "reward_implicit": self.reward_implicit,
            "generated_response": self.generated_response,
            "metadata": self.original.metadata or {},
        }


class DatasetLoader:
    """Load and manage toxicity datasets from HuggingFace."""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the dataset loader.

        Args:
            settings: Configuration settings. If None, loads from environment.
        """
        self.settings = settings or Settings()
        self._dataset: Optional[DatasetDict | Dataset] = None

    def load(self, dataset_name: Optional[str] = None, split: Optional[str] = None) -> Dataset:
        """Load dataset from HuggingFace.

        Args:
            dataset_name: HuggingFace dataset identifier. Defaults to config value.
            split: Dataset split to load. Defaults to config value.

        Returns:
            Loaded dataset.
        """
        name = dataset_name or self.settings.dataset_name
        split = split or self.settings.dataset_split

        logger.info(f"Loading dataset: {name}, split: {split}")

        try:
            self._dataset = load_dataset(name, split=split)
            logger.info(f"Loaded {len(self._dataset)} samples")
            return self._dataset
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise DatasetError(f"Failed to load dataset '{name}': {e}") from e

    def get_column_names(self) -> list[str]:
        """Get column names from loaded dataset."""
        if self._dataset is None:
            raise DatasetError("Dataset not loaded. Call load() first.")
        return self._dataset.column_names

    def infer_text_column(self) -> str:
        """Infer the text column name from the dataset.

        Returns:
            The most likely text column name.

        Raises:
            DatasetError: If dataset not loaded or text column cannot be inferred.
        """
        if self._dataset is None:
            raise DatasetError("Dataset not loaded. Call load() first.")

        columns = self.get_column_names()

        # Common text column names
        text_candidates = ["text", "content", "sentence", "comment", "message", "input"]
        for candidate in text_candidates:
            if candidate in columns:
                return candidate

        # Fall back to first string column
        for col in columns:
            sample = self._dataset[0][col]
            if isinstance(sample, str):
                return col

        raise DatasetError(f"Could not infer text column from: {columns}")

    def infer_label_column(self) -> Optional[str]:
        """Infer the label column name from the dataset.

        Returns:
            The most likely label column name, or None if not found.

        Raises:
            DatasetError: If dataset not loaded.
        """
        if self._dataset is None:
            raise DatasetError("Dataset not loaded. Call load() first.")

        columns = self.get_column_names()

        # Common label column names
        label_candidates = ["label", "toxic", "toxicity", "is_toxic", "class", "target"]
        for candidate in label_candidates:
            if candidate in columns:
                return candidate

        return None

    def iter_samples(
        self,
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Iterator[ToxicitySample]:
        """Iterate over samples in the dataset.

        Args:
            text_column: Name of the text column. If None, will be inferred.
            label_column: Name of the label column. If None, will be inferred.
            limit: Maximum number of samples to yield.

        Yields:
            ToxicitySample objects.

        Raises:
            DatasetError: If dataset not loaded.
        """
        if self._dataset is None:
            raise DatasetError("Dataset not loaded. Call load() first.")

        text_col = text_column or self.infer_text_column()
        label_col = label_column or self.infer_label_column()

        logger.info(f"Using text column: {text_col}, label column: {label_col}")

        for idx, sample in enumerate(self._dataset):
            if limit and idx >= limit:
                break

            text = sample[text_col]
            label = sample.get(label_col) if label_col else None

            # Collect other columns as metadata
            metadata = {
                k: v for k, v in sample.items() if k not in [text_col, label_col]
            }

            yield ToxicitySample(
                id=str(idx),
                text=text,
                label=label,
                metadata=metadata if metadata else None,
            )

    def to_dataframe(
        self,
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Convert dataset to pandas DataFrame.

        Args:
            text_column: Name of the text column.
            label_column: Name of the label column.
            limit: Maximum number of samples.

        Returns:
            DataFrame with samples.
        """
        samples = list(self.iter_samples(text_column, label_column, limit))
        return pd.DataFrame([s.to_dict() for s in samples])

    def save_augmented(
        self,
        samples: list[AugmentedSample],
        output_path: Optional[str] = None,
        format: str = "parquet",
    ) -> Path:
        """Save augmented samples to disk.

        Args:
            samples: List of augmented samples to save.
            output_path: Output file path. Defaults to config output_dir.
            format: Output format ('parquet', 'json', 'csv').

        Returns:
            Path to saved file.
        """
        output_dir = Path(output_path or self.settings.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame([s.to_dict() for s in samples])

        if format == "parquet":
            path = output_dir / "augmented_dataset.parquet"
            df.to_parquet(path, index=False)
        elif format == "json":
            path = output_dir / "augmented_dataset.json"
            df.to_json(path, orient="records", indent=2)
        elif format == "csv":
            path = output_dir / "augmented_dataset.csv"
            df.to_csv(path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Saved {len(samples)} augmented samples to {path}")
        return path

    def load_augmented(self, path: str | Path) -> list[AugmentedSample]:
        """Load previously augmented samples from disk.

        Args:
            path: Path to saved file.

        Returns:
            List of augmented samples.
        """
        path = Path(path)

        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".json":
            df = pd.read_json(path)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        samples = []
        for _, row in df.iterrows():
            original = ToxicitySample(
                id=row["id"],
                text=row["text"],
                label=row.get("label"),
                metadata=row.get("metadata"),
            )
            samples.append(
                AugmentedSample(
                    original=original,
                    rubrics=row["rubrics"],
                    reward_explicit=row.get("reward_explicit"),
                    reward_implicit=row.get("reward_implicit"),
                    generated_response=row.get("generated_response"),
                )
            )

        logger.info(f"Loaded {len(samples)} augmented samples from {path}")
        return samples
