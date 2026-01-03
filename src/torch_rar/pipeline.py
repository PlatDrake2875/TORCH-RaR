"""Main pipeline for dataset augmentation using RaR method."""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from torch_rar.config import Settings
from torch_rar.data_loader import AugmentedSample, DatasetLoader, ToxicitySample
from torch_rar.exceptions import TorchRarError, ValidationError
from torch_rar.llm_client import LLMClient
from torch_rar.logging_config import get_logger
from torch_rar.progress import ProgressTracker
from torch_rar.prompt_templates import PromptTemplateRegistry
from torch_rar.reward_calculator import RewardCalculator
from torch_rar.rubric_generator import RubricGenerator

logger = get_logger(__name__)


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

        # Create shared template registry for rubric generator and reward calculator
        self.template_registry = PromptTemplateRegistry(
            self.settings.prompt_templates.directory
        )

        self.rubric_generator = RubricGenerator(
            self.settings, self.llm_client, self.template_registry
        )
        self.reward_calculator = RewardCalculator(
            self.settings, self.llm_client, self.template_registry
        )

        # Progress tracking
        self.progress = ProgressTracker(disable=not self.settings.show_progress)

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

        # Reset progress stats for this run
        self.progress.reset_stats()

        # Process with Rich progress bar
        batch_size = self.settings.batch_size
        num_batches = (total_samples + batch_size - 1) // batch_size

        with self.progress.track_pipeline(
            total=num_batches,
            description="[cyan]Augmenting dataset",
        ) as tracker:
            for i in range(0, total_samples, batch_size):
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
                        if tracker:
                            tracker.fail(1)
                    elif result is None:
                        if tracker:
                            tracker.fail(1)
                    else:
                        augmented_samples.append(result)

                # Advance progress after processing batch
                if tracker:
                    tracker.advance(1)

        # Get final progress stats
        progress_stats = self.progress.get_stats()
        failed_count = progress_stats["failed"]

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
        logger.info("Pipeline Statistics:")
        logger.info(f"  Total samples: {stats.total_samples}")
        logger.info(f"  Processed: {stats.processed_samples}")
        logger.info(f"  Failed: {stats.failed_samples}")
        logger.info(f"  Avg rubrics/sample: {stats.avg_rubrics_per_sample:.2f}")
        logger.info(f"  Avg explicit reward: {stats.avg_explicit_reward:.4f}")
        if stats.avg_implicit_reward is not None:
            logger.info(f"  Avg implicit reward: {stats.avg_implicit_reward:.4f}")
        logger.info(f"  Execution time: {stats.execution_time_seconds:.2f}s")

        # Log cache statistics
        cache_stats = self.rubric_generator.get_cache_stats()
        logger.info("Cache Statistics:")
        logger.info(f"  Hits: {cache_stats['hits']}")
        logger.info(f"  Misses: {cache_stats['misses']}")
        logger.info(f"  Hit rate: {cache_stats['hit_rate']}")
        if "cache_size_mb" in cache_stats:
            logger.info(f"  Cache size: {cache_stats['cache_size_mb']:.2f} MB")

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
