"""Main pipeline for dataset augmentation using RaR method."""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from tqdm import tqdm

from torch_rar.config import Settings
from torch_rar.data_loader import AugmentedSample, DatasetLoader, ToxicitySample
from torch_rar.exceptions import TorchRarError, ValidationError
from torch_rar.llm_client import LLMClient
from torch_rar.reward_calculator import RewardCalculator
from torch_rar.rubric_generator import RubricGenerator

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Statistics from pipeline execution."""

    total_samples: int
    processed_samples: int
    failed_samples: int
    avg_rubrics_per_sample: float
    avg_explicit_reward: float
    avg_implicit_reward: Optional[float]
    execution_time_seconds: float


class AugmentationPipeline:
    """Pipeline for augmenting toxicity datasets using Rubrics as Rewards."""

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize the pipeline.

        Args:
            settings: Configuration settings.
        """
        self.settings = settings or Settings()
        self.llm_client = LLMClient(self.settings)
        self.data_loader = DatasetLoader(self.settings)
        self.rubric_generator = RubricGenerator(self.settings, self.llm_client)
        self.reward_calculator = RewardCalculator(self.settings, self.llm_client)

    async def process_sample(
        self,
        sample: ToxicitySample,
        use_predefined_rubrics: bool = False,
        reward_method: str = "both",
    ) -> Optional[AugmentedSample]:
        """Process a single sample through the RaR pipeline.

        Args:
            sample: The toxicity sample to process.
            use_predefined_rubrics: If True, use predefined static rubrics.
            reward_method: "explicit", "implicit", or "both".

        Returns:
            AugmentedSample with rubrics and rewards, or None if processing failed.
        """
        # Input validation
        if not sample.text or not sample.text.strip():
            logger.warning(f"Sample {sample.id} has empty text, skipping")
            return None

        try:
            # Generate or use predefined rubrics
            if use_predefined_rubrics:
                rubrics = self.rubric_generator.get_predefined_rubrics()
            else:
                rubrics = await self.rubric_generator.generate_rubrics(sample.text)

            if not rubrics:
                logger.warning(f"No rubrics generated for sample {sample.id}")
                return None

            # Calculate rewards
            reward_result = await self.reward_calculator.calculate_reward(
                text=sample.text,
                rubrics=rubrics,
                method=reward_method,
            )

            return AugmentedSample(
                original=sample,
                rubrics=[r.to_dict() for r in rubrics],
                reward_explicit=reward_result.explicit_reward,
                reward_implicit=reward_result.implicit_reward,
            )

        except ValidationError as e:
            logger.warning(f"Validation error for sample {sample.id}: {e}")
            return None
        except TorchRarError as e:
            logger.error(f"Processing error for sample {sample.id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error processing sample {sample.id}: {e}")
            return None

    async def run(
        self,
        limit: Optional[int] = None,
        use_predefined_rubrics: bool = False,
        reward_method: str = "both",
        output_format: str = "parquet",
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
    ) -> tuple[list[AugmentedSample], PipelineStats]:
        """Run the full augmentation pipeline.

        Args:
            limit: Maximum number of samples to process.
            use_predefined_rubrics: If True, use predefined static rubrics.
            reward_method: "explicit", "implicit", or "both".
            output_format: Output format for saving results.
            text_column: Name of text column in dataset.
            label_column: Name of label column in dataset.

        Returns:
            Tuple of (list of augmented samples, pipeline statistics).
        """
        start_time = datetime.now()

        # Load dataset
        logger.info("Loading dataset...")
        self.data_loader.load()

        # Get samples
        samples = list(
            self.data_loader.iter_samples(
                text_column=text_column,
                label_column=label_column,
                limit=limit,
            )
        )
        total_samples = len(samples)
        logger.info(f"Processing {total_samples} samples...")

        # Process samples in batches
        augmented_samples: list[AugmentedSample] = []
        failed_count = 0

        # Process with progress bar
        batch_size = self.settings.batch_size
        for i in tqdm(range(0, total_samples, batch_size), desc="Processing batches"):
            batch = samples[i : i + batch_size]

            # Process batch concurrently
            tasks = [
                self.process_sample(s, use_predefined_rubrics, reward_method)
                for s in batch
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Batch processing error: {result}")
                    failed_count += 1
                elif result is None:
                    failed_count += 1
                else:
                    augmented_samples.append(result)

        # Calculate statistics
        execution_time = (datetime.now() - start_time).total_seconds()

        total_rubrics = sum(len(s.rubrics) for s in augmented_samples)
        avg_rubrics = total_rubrics / len(augmented_samples) if augmented_samples else 0

        explicit_rewards = [s.reward_explicit for s in augmented_samples if s.reward_explicit]
        avg_explicit = sum(explicit_rewards) / len(explicit_rewards) if explicit_rewards else 0

        implicit_rewards = [
            s.reward_implicit for s in augmented_samples if s.reward_implicit is not None
        ]
        avg_implicit = (
            sum(implicit_rewards) / len(implicit_rewards) if implicit_rewards else None
        )

        stats = PipelineStats(
            total_samples=total_samples,
            processed_samples=len(augmented_samples),
            failed_samples=failed_count,
            avg_rubrics_per_sample=avg_rubrics,
            avg_explicit_reward=avg_explicit,
            avg_implicit_reward=avg_implicit,
            execution_time_seconds=execution_time,
        )

        # Save results
        if augmented_samples:
            output_path = self.data_loader.save_augmented(
                augmented_samples, format=output_format
            )
            logger.info(f"Saved augmented dataset to {output_path}")

        # Log statistics
        logger.info(f"Pipeline Statistics:")
        logger.info(f"  Total samples: {stats.total_samples}")
        logger.info(f"  Processed: {stats.processed_samples}")
        logger.info(f"  Failed: {stats.failed_samples}")
        logger.info(f"  Avg rubrics/sample: {stats.avg_rubrics_per_sample:.2f}")
        logger.info(f"  Avg explicit reward: {stats.avg_explicit_reward:.4f}")
        if stats.avg_implicit_reward is not None:
            logger.info(f"  Avg implicit reward: {stats.avg_implicit_reward:.4f}")
        logger.info(f"  Execution time: {stats.execution_time_seconds:.2f}s")

        return augmented_samples, stats


def run_pipeline(
    limit: Optional[int] = None,
    use_predefined_rubrics: bool = False,
    reward_method: str = "both",
    output_format: str = "parquet",
) -> tuple[list[AugmentedSample], PipelineStats]:
    """Convenience function to run the pipeline synchronously.

    Args:
        limit: Maximum number of samples to process.
        use_predefined_rubrics: If True, use predefined static rubrics.
        reward_method: "explicit", "implicit", or "both".
        output_format: Output format for saving results.

    Returns:
        Tuple of (list of augmented samples, pipeline statistics).
    """
    pipeline = AugmentationPipeline()
    return asyncio.run(
        pipeline.run(
            limit=limit,
            use_predefined_rubrics=use_predefined_rubrics,
            reward_method=reward_method,
            output_format=output_format,
        )
    )
