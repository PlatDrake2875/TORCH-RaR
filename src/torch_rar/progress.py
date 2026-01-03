"""Progress tracking with Rich for the TORCH-RaR pipeline.

Features:
- Rich progress bars with detailed stats
- Nested progress support for batch operations
- Statistics collection (processed, failed, cached)
- Async-compatible progress tracking
"""

from contextlib import contextmanager
from typing import Iterator, Optional

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console()


class ProgressContext:
    """Context for tracking progress during pipeline execution."""

    def __init__(self, progress: Progress, task_id, stats: dict):
        self._progress = progress
        self._task_id = task_id
        self._stats = stats

    def advance(self, n: int = 1, cached: bool = False) -> None:
        """Advance progress by n steps.

        Args:
            n: Number of steps to advance.
            cached: Whether these steps used cached data.
        """
        self._progress.advance(self._task_id, n)
        self._stats["processed"] += n
        if cached:
            self._stats["cached"] += n

    def fail(self, n: int = 1) -> None:
        """Record n failures.

        Args:
            n: Number of failures to record.
        """
        self._stats["failed"] += n

    def update_description(self, desc: str) -> None:
        """Update the progress bar description.

        Args:
            desc: New description text.
        """
        self._progress.update(self._task_id, description=desc)


class ProgressTracker:
    """Unified progress tracking for sync and async operations.

    Features:
    - Rich progress bars with detailed stats
    - Statistics collection
    - Disable mode for non-interactive use
    """

    def __init__(self, disable: bool = False):
        """Initialize the progress tracker.

        Args:
            disable: If True, disable all progress display.
        """
        self.disable = disable
        self._stats = {
            "processed": 0,
            "failed": 0,
            "cached": 0,
        }

    @contextmanager
    def track_pipeline(
        self,
        total: int,
        description: str = "Processing",
    ) -> Iterator[Optional[ProgressContext]]:
        """Context manager for pipeline-level progress.

        Args:
            total: Total number of items to process.
            description: Description for the progress bar.

        Yields:
            ProgressContext for tracking progress, or None if disabled.
        """
        if self.disable:
            yield None
            return

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            refresh_per_second=2,
        ) as progress:
            task_id = progress.add_task(description, total=total)
            yield ProgressContext(progress, task_id, self._stats)

    def get_stats(self) -> dict:
        """Return progress statistics.

        Returns:
            Dictionary with processed, failed, cached counts and rates.
        """
        total = self._stats["processed"]
        success = total - self._stats["failed"]

        return {
            **self._stats,
            "success_rate": (
                f"{success / total:.2%}" if total > 0 else "N/A"
            ),
            "cache_rate": (
                f"{self._stats['cached'] / total:.2%}" if total > 0 else "N/A"
            ),
        }

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self._stats = {
            "processed": 0,
            "failed": 0,
            "cached": 0,
        }
