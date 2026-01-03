"""Loguru-based logging configuration for TORCH-RaR.

Features:
- Colored console output for development
- JSON-formatted file logs for production/analysis
- Automatic log rotation and retention
- Context binding for structured logging
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field


class LoggingConfig(BaseModel):
    """Configuration for logging settings."""

    level: str = Field(
        default="INFO",
        description="Log level (DEBUG, INFO, WARNING, ERROR)",
    )
    directory: str = Field(
        default="logs",
        description="Directory for log files",
    )
    json_format: bool = Field(
        default=True,
        description="Whether to output JSON to files",
    )
    rotation: str = Field(
        default="10 MB",
        description="When to rotate log files",
    )
    retention: str = Field(
        default="7 days",
        description="How long to keep old logs",
    )


def setup_logging(
    config: Optional[LoggingConfig] = None,
    verbose: bool = False,
) -> None:
    """Configure logging for the TORCH-RaR pipeline.

    Args:
        config: LoggingConfig instance with settings. Uses defaults if None.
        verbose: If True, overrides config level to DEBUG.
    """
    if config is None:
        config = LoggingConfig()

    # Determine log level
    level = "DEBUG" if verbose else config.level

    # Create log directory
    log_path = Path(config.directory)
    log_path.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Console handler - human-readable, colored
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        level=level,
        colorize=True,
    )

    # File handler - JSON or plain text
    timestamp = datetime.now().strftime("%Y%m%d")

    if config.json_format:
        logger.add(
            log_path / f"pipeline_{timestamp}.json",
            format="{message}",
            level=level,
            serialize=True,
            rotation=config.rotation,
            retention=config.retention,
            compression="gz",
        )
    else:
        logger.add(
            log_path / f"pipeline_{timestamp}.log",
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                "{name}:{function}:{line} | {message}"
            ),
            level=level,
            rotation=config.rotation,
            retention=config.retention,
            compression="gz",
        )

    # Error-only file for quick issue identification
    logger.add(
        log_path / "errors.log",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | {level} | "
            "{name}:{function}:{line}\n{message}\n{exception}"
        ),
        level="ERROR",
        rotation="5 MB",
        retention="30 days",
    )

    logger.info(
        "Logging initialized",
        log_dir=str(log_path),
        level=level,
    )


def get_logger(name: str):
    """Get a contextualized logger for a module.

    Usage:
        logger = get_logger(__name__)
        logger.info("Processing sample", sample_id=123)

    Args:
        name: Module name (typically __name__)

    Returns:
        Logger instance bound with module context
    """
    return logger.bind(module=name)
