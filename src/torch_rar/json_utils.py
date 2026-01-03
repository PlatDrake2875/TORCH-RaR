"""JSON parsing utilities for LLM responses.

This module provides robust JSON extraction and parsing from LLM responses,
handling common response formats like markdown code blocks.
"""

import json
import re
from typing import Any

from torch_rar.exceptions import JSONParseError


def extract_json_from_response(
    response: str,
    expected_type: str = "object",
) -> Any:
    """Extract and parse JSON from LLM response text.

    Handles common LLM response formats:
    - Markdown code blocks (```json ... ```)
    - Plain code blocks (``` ... ```)
    - Raw JSON objects/arrays

    Args:
        response: Raw LLM response text.
        expected_type: Expected JSON type - "object" for {} or "array" for [].

    Returns:
        Parsed JSON data (dict for object, list for array).

    Raises:
        JSONParseError: If JSON cannot be extracted or parsed.

    Examples:
        >>> extract_json_from_response('{"key": "value"}')
        {'key': 'value'}

        >>> extract_json_from_response('```json\\n[1, 2, 3]\\n```', expected_type="array")
        [1, 2, 3]
    """
    if not response:
        raise JSONParseError("Empty response cannot be parsed as JSON")

    text = response.strip()

    # Handle markdown code blocks (```json ... ```)
    if "```json" in text:
        start = text.find("```json") + 7
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()
    # Handle plain code blocks (``` ... ```)
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()

    # Extract JSON structure based on expected type
    if expected_type == "array":
        if "[" not in text:
            raise JSONParseError("Expected JSON array but no '[' found in response")
        start = text.find("[")
        end = text.rfind("]")
        if end <= start:
            raise JSONParseError("Malformed JSON array: missing closing ']'")
        text = text[start : end + 1]
    elif expected_type == "object":
        if "{" not in text:
            raise JSONParseError("Expected JSON object but no '{' found in response")
        start = text.find("{")
        end = text.rfind("}")
        if end <= start:
            raise JSONParseError("Malformed JSON object: missing closing '}'")
        text = text[start : end + 1]

    try:
        return json.loads(text.strip())
    except json.JSONDecodeError as e:
        raise JSONParseError(f"Failed to parse JSON: {e}") from e


def extract_boolean_from_response(response: str) -> bool:
    """Extract a boolean value from LLM response text.

    Useful for simple yes/no evaluations where JSON parsing may be overkill.

    Args:
        response: Raw LLM response text.

    Returns:
        True if response indicates satisfaction/true, False otherwise.
    """
    lower = response.lower()

    # Check for explicit true/false
    if '"satisfied": true' in lower or '"satisfied":true' in lower:
        return True
    if '"satisfied": false' in lower or '"satisfied":false' in lower:
        return False

    # Check for text indicators
    positive_indicators = ["satisfied", "true", "yes", "correct", "met"]
    negative_indicators = ["not satisfied", "false", "no", "incorrect", "not met"]

    for neg in negative_indicators:
        if neg in lower:
            return False

    for pos in positive_indicators:
        if pos in lower:
            return True

    return False


def extract_rating_from_response(response: str, min_val: int = 1, max_val: int = 10) -> int:
    """Extract a numeric rating from LLM response text.

    Args:
        response: Raw LLM response text.
        min_val: Minimum valid rating value.
        max_val: Maximum valid rating value.

    Returns:
        Extracted rating clamped to [min_val, max_val].
        Defaults to middle value if no rating found.
    """
    # Try JSON extraction first
    try:
        data = extract_json_from_response(response, expected_type="object")
        if isinstance(data, dict) and "rating" in data:
            rating = int(data["rating"])
            return max(min_val, min(max_val, rating))
    except (JSONParseError, ValueError, TypeError):
        pass

    # Fall back to regex extraction
    pattern = rf"\b({min_val}|[{min_val + 1}-{max_val - 1}]|{max_val})\b"
    numbers = re.findall(pattern, response)
    if numbers:
        rating = int(numbers[0])
        return max(min_val, min(max_val, rating))

    # Default to middle value
    return (min_val + max_val) // 2
