"""Custom exceptions for TORCH-RaR.

This module defines a hierarchy of exceptions for better error handling
and debugging throughout the TORCH-RaR pipeline.
"""


class TorchRarError(Exception):
    """Base exception for all TORCH-RaR errors.

    All custom exceptions in this package inherit from this class,
    allowing callers to catch all TORCH-RaR errors with a single handler.
    """

    pass


class ConfigurationError(TorchRarError):
    """Error in configuration settings.

    Raised when:
    - Settings file is missing or malformed
    - Required configuration values are not provided
    - Invalid configuration values are specified
    """

    pass


class LLMClientError(TorchRarError):
    """Error during LLM API communication.

    Raised when:
    - API requests fail after retries
    - Authentication errors occur
    - Rate limits are exceeded beyond retry capacity
    """

    pass


class JSONParseError(TorchRarError):
    """Error parsing JSON from LLM response.

    Raised when:
    - LLM response doesn't contain valid JSON
    - JSON structure doesn't match expected format
    - Markdown code block extraction fails
    """

    pass


class RubricGenerationError(TorchRarError):
    """Error during rubric generation.

    Raised when:
    - LLM fails to generate valid rubrics
    - Generated rubrics don't meet validation criteria
    - Input text is invalid for rubric generation
    """

    pass


class RewardCalculationError(TorchRarError):
    """Error during reward calculation.

    Raised when:
    - Rubric evaluation fails
    - Reward aggregation encounters invalid data
    - Input validation fails for reward calculation
    """

    pass


class DatasetError(TorchRarError):
    """Error loading or processing dataset.

    Raised when:
    - HuggingFace dataset fails to load
    - Dataset columns cannot be inferred
    - Data format is invalid or corrupted
    """

    pass


class ValidationError(TorchRarError):
    """Error during input validation.

    Raised when:
    - Input text is empty or exceeds limits
    - Required parameters are missing
    - Parameter values are out of valid range
    """

    pass
