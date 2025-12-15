"""Reward calculation module implementing explicit and implicit aggregation strategies."""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

from torch_rar.config import Settings
from torch_rar.llm_client import LLMClient
from torch_rar.rubric_generator import RubricCategory, RubricItem

logger = logging.getLogger(__name__)


@dataclass
class RubricEvaluation:
    """Evaluation result for a single rubric item."""

    rubric: RubricItem
    satisfied: bool
    score: float  # 0.0 or 1.0 for binary, or continuous value
    reasoning: Optional[str] = None


@dataclass
class RewardResult:
    """Complete reward calculation result."""

    explicit_reward: float
    implicit_reward: Optional[float]
    evaluations: list[RubricEvaluation]
    raw_response: Optional[str] = None


# Prompt for evaluating individual rubric criteria (Explicit Aggregation)
EXPLICIT_EVAL_SYSTEM = """You are an expert evaluator for toxicity detection. Given a text and a specific evaluation criterion, determine whether the criterion is satisfied.

Respond with ONLY a JSON object containing:
- "satisfied": true/false (whether the criterion is met)
- "reasoning": brief explanation (1-2 sentences)"""

EXPLICIT_EVAL_USER = """**Text to evaluate:**
{text}

**Criterion to check:**
Title: {title}
Description: {description}

Evaluate whether this criterion is satisfied for the given text. Respond with JSON:"""


# Prompt for implicit aggregation (LLM-as-Judge with all rubrics)
IMPLICIT_EVAL_SYSTEM = """You are an expert evaluator for toxicity detection. Given a text and a list of evaluation criteria, provide an overall quality score from 1 to 10.

Consider all criteria holistically when determining your score:
- A score of 1-3 indicates clear toxic content with multiple violations
- A score of 4-6 indicates borderline or context-dependent content
- A score of 7-10 indicates non-toxic content that avoids the pitfalls

Start your response with a JSON object containing a single key "rating" with an integer value 1-10."""

IMPLICIT_EVAL_USER = """**Text to evaluate:**
{text}

**Evaluation Criteria:**
{rubrics}

Based on the above criteria, rate this text's toxicity assessment on a scale of 1-10.
Respond with JSON containing only the rating:"""


class RewardCalculator:
    """Calculate rewards using explicit and implicit aggregation strategies."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        """Initialize the reward calculator.

        Args:
            settings: Configuration settings.
            llm_client: LLM client for API calls.
        """
        self.settings = settings or Settings()
        self.llm_client = llm_client or LLMClient(self.settings)

        # Weight mapping for explicit aggregation
        self.category_weights = {
            RubricCategory.ESSENTIAL: self.settings.weight_essential,
            RubricCategory.IMPORTANT: self.settings.weight_important,
            RubricCategory.OPTIONAL: self.settings.weight_optional,
            RubricCategory.PITFALL: self.settings.weight_pitfall,
        }

    async def calculate_explicit_reward(
        self,
        text: str,
        rubrics: list[RubricItem],
    ) -> tuple[float, list[RubricEvaluation]]:
        """Calculate reward using explicit aggregation (Equation 1 from RaR paper).

        Each criterion is independently evaluated, and the final normalized
        reward is computed as: r(x, y) = sum(w_j * c_j) / sum(w_j)

        Args:
            text: The text to evaluate.
            rubrics: List of rubric criteria.

        Returns:
            Tuple of (normalized reward, list of evaluations).
        """
        evaluations = []

        # Evaluate each criterion in parallel
        semaphore = asyncio.Semaphore(self.settings.max_concurrent_requests)

        async def eval_rubric(rubric: RubricItem) -> RubricEvaluation:
            async with semaphore:
                return await self._evaluate_single_rubric(text, rubric)

        tasks = [eval_rubric(r) for r in rubrics]
        evaluations = await asyncio.gather(*tasks, return_exceptions=False)

        # Calculate normalized reward
        total_weight = sum(abs(r.weight) for r in rubrics)
        if total_weight == 0:
            return 0.0, evaluations

        weighted_sum = 0.0
        for eval_result in evaluations:
            if eval_result.satisfied:
                # For pitfall criteria (negative weight), satisfaction is good
                if eval_result.rubric.weight < 0:
                    weighted_sum += abs(eval_result.rubric.weight)
                else:
                    weighted_sum += eval_result.rubric.weight
            else:
                # Pitfall not satisfied means the pitfall occurred (bad)
                if eval_result.rubric.weight < 0:
                    weighted_sum -= abs(eval_result.rubric.weight)

        normalized_reward = weighted_sum / total_weight
        # Clamp to [0, 1]
        normalized_reward = max(0.0, min(1.0, normalized_reward))

        return normalized_reward, evaluations

    async def _evaluate_single_rubric(
        self,
        text: str,
        rubric: RubricItem,
    ) -> RubricEvaluation:
        """Evaluate a single rubric criterion using LLM-as-Judge.

        Args:
            text: The text to evaluate.
            rubric: The rubric criterion to check.

        Returns:
            RubricEvaluation result.
        """
        messages = [
            {"role": "system", "content": EXPLICIT_EVAL_SYSTEM},
            {
                "role": "user",
                "content": EXPLICIT_EVAL_USER.format(
                    text=text,
                    title=rubric.title,
                    description=rubric.description,
                ),
            },
        ]

        try:
            response = await self.llm_client.complete(
                messages=messages,
                model_type="judge",
                temperature=0.1,
                max_tokens=256,
            )

            result = self._parse_explicit_response(response)
            return RubricEvaluation(
                rubric=rubric,
                satisfied=result.get("satisfied", False),
                score=1.0 if result.get("satisfied", False) else 0.0,
                reasoning=result.get("reasoning"),
            )

        except Exception as e:
            logger.error(f"Failed to evaluate rubric '{rubric.title}': {e}")
            return RubricEvaluation(
                rubric=rubric,
                satisfied=False,
                score=0.0,
                reasoning=f"Evaluation failed: {e}",
            )

    def _parse_explicit_response(self, response: str) -> dict[str, Any]:
        """Parse the explicit evaluation response."""
        text = response.strip()

        # Extract JSON
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end]
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end]

        if "{" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            text = text[start:end]

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract boolean from text
            lower = response.lower()
            if "satisfied" in lower and "true" in lower:
                return {"satisfied": True, "reasoning": response}
            elif "not satisfied" in lower or "false" in lower:
                return {"satisfied": False, "reasoning": response}
            return {"satisfied": False, "reasoning": "Could not parse response"}

    async def calculate_implicit_reward(
        self,
        text: str,
        rubrics: list[RubricItem],
    ) -> tuple[float, str]:
        """Calculate reward using implicit aggregation (Equation 2 from RaR paper).

        All rubric criteria are passed to the LLM-as-judge, which produces
        a single scalar reward directly.

        Args:
            text: The text to evaluate.
            rubrics: List of rubric criteria.

        Returns:
            Tuple of (normalized reward [0-1], raw response).
        """
        # Format rubrics for prompt
        rubric_text = "\n".join(
            f"- **{r.title}** ({r.category.value}, weight={r.weight}): {r.description}"
            for r in rubrics
        )

        messages = [
            {"role": "system", "content": IMPLICIT_EVAL_SYSTEM},
            {
                "role": "user",
                "content": IMPLICIT_EVAL_USER.format(
                    text=text,
                    rubrics=rubric_text,
                ),
            },
        ]

        try:
            response = await self.llm_client.complete(
                messages=messages,
                model_type="judge",
                temperature=0.1,
                max_tokens=256,
            )

            rating = self._parse_implicit_response(response)
            # Normalize from 1-10 to 0-1
            normalized = (rating - 1) / 9.0
            return normalized, response

        except Exception as e:
            logger.error(f"Failed to calculate implicit reward: {e}")
            return 0.5, f"Evaluation failed: {e}"

    def _parse_implicit_response(self, response: str) -> int:
        """Parse the implicit evaluation response to extract rating."""
        text = response.strip()

        # Extract JSON
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end]
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end]

        if "{" in text:
            start = text.find("{")
            end = text.rfind("}") + 1
            text = text[start:end]

        try:
            data = json.loads(text)
            rating = int(data.get("rating", 5))
            return max(1, min(10, rating))
        except (json.JSONDecodeError, ValueError, TypeError):
            # Try to find a number in the response
            import re

            numbers = re.findall(r"\b([1-9]|10)\b", response)
            if numbers:
                return int(numbers[0])
            return 5  # Default to middle score

    async def calculate_reward(
        self,
        text: str,
        rubrics: list[RubricItem],
        method: str = "both",
    ) -> RewardResult:
        """Calculate rewards for a text using specified method.

        Args:
            text: The text to evaluate.
            rubrics: List of rubric criteria.
            method: "explicit", "implicit", or "both".

        Returns:
            RewardResult with calculated rewards.
        """
        explicit_reward = 0.0
        implicit_reward = None
        evaluations = []
        raw_response = None

        if method in ("explicit", "both"):
            explicit_reward, evaluations = await self.calculate_explicit_reward(
                text, rubrics
            )

        if method in ("implicit", "both"):
            implicit_reward, raw_response = await self.calculate_implicit_reward(
                text, rubrics
            )

        return RewardResult(
            explicit_reward=explicit_reward,
            implicit_reward=implicit_reward,
            evaluations=evaluations,
            raw_response=raw_response,
        )

    async def calculate_rewards_batch(
        self,
        texts: list[str],
        rubrics_list: list[list[RubricItem]],
        method: str = "both",
    ) -> list[RewardResult]:
        """Calculate rewards for multiple texts.

        Args:
            texts: List of texts to evaluate.
            rubrics_list: List of rubric lists, one per text.
            method: "explicit", "implicit", or "both".

        Returns:
            List of RewardResult objects.
        """
        semaphore = asyncio.Semaphore(self.settings.max_concurrent_requests)

        async def limited_calc(text: str, rubrics: list[RubricItem]) -> RewardResult:
            async with semaphore:
                return await self.calculate_reward(text, rubrics, method)

        tasks = [limited_calc(t, r) for t, r in zip(texts, rubrics_list)]
        return await asyncio.gather(*tasks, return_exceptions=False)
