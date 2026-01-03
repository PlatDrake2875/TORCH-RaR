"""Reward calculation module implementing explicit and implicit aggregation strategies.

This module implements the two reward aggregation methods from the RaR paper:

1. EXPLICIT AGGREGATION (Equation 1 from RaR paper):
   r(x, ŷ) = Σⱼ(wⱼ · cⱼ(x, ŷ)) / Σⱼ|wⱼ|

   Where:
   - wⱼ is the weight for criterion j (positive for E/I, negative for P)
   - cⱼ(x, ŷ) ∈ {0, 1} indicates if criterion j is satisfied

   For TORCH-RaR:
   - Essential (E1-E4): w = 1.0, satisfied when toxicity indicator is DETECTED
   - Important (I1-I4): w = 0.7, satisfied when context is properly CONSIDERED
   - Pitfall (P1-P3): w = -0.9, satisfied when error is AVOIDED

2. IMPLICIT AGGREGATION (Equation 2 from RaR paper):
   All rubrics are passed to the LLM judge for holistic scoring (1-10 Likert).
   The judge considers all criteria and provides a single scalar reward.

The prompts are designed for Romanian political discourse evaluation.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Optional

from torch_rar.config import Settings
from torch_rar.exceptions import JSONParseError, RewardCalculationError, ValidationError
from torch_rar.json_utils import (
    extract_boolean_from_response,
    extract_json_from_response,
    extract_rating_from_response,
)
from torch_rar.llm_client import LLMClient
from torch_rar.rubric_generator import RubricCategory, RubricItem

logger = logging.getLogger(__name__)


@dataclass
class RubricEvaluation:
    """Evaluation result for a single rubric item.

    For TORCH-RaR rubrics:
    - Essential (E1-E4): satisfied=True means toxicity indicator was DETECTED
    - Important (I1-I4): satisfied=True means context was properly CONSIDERED
    - Pitfall (P1-P3): satisfied=True means the error was AVOIDED

    Attributes:
        rubric: The RubricItem being evaluated
        rubric_id: Quick reference ID (e.g., "E1", "P2")
        satisfied: Whether the criterion is satisfied
        score: Numerical score (typically 0.0 or 1.0)
        reasoning: LLM's explanation for the evaluation
    """

    rubric: RubricItem
    satisfied: bool
    score: float  # 0.0 or 1.0 for binary, or continuous value
    rubric_id: str = ""
    reasoning: Optional[str] = None

    def __post_init__(self):
        """Set rubric_id from rubric if not provided."""
        if not self.rubric_id and self.rubric:
            self.rubric_id = self.rubric.rubric_id


@dataclass
class RewardResult:
    """Complete reward calculation result following RaR methodology.

    Contains both explicit and implicit reward scores, along with
    detailed per-rubric evaluations for interpretability.

    Attributes:
        explicit_reward: Normalized weighted sum reward [0, 1]
        implicit_reward: LLM judge holistic score [0, 1] (normalized from 1-10)
        evaluations: List of per-rubric evaluation results
        raw_response: Raw LLM response for implicit evaluation
        category_scores: Breakdown by rubric category (E, I, P)
    """

    explicit_reward: float
    implicit_reward: Optional[float]
    evaluations: list[RubricEvaluation]
    raw_response: Optional[str] = None
    category_scores: Optional[dict[str, float]] = None


# =============================================================================
# EXPLICIT AGGREGATION PROMPTS
# =============================================================================
# These prompts evaluate each rubric criterion independently.
# The LLM acts as a judge for each criterion (E1-E4, I1-I4, P1-P3).
# =============================================================================

EXPLICIT_EVAL_SYSTEM = """You are an expert evaluator for toxicity detection in Romanian political discourse.

Your task is to evaluate whether a specific criterion is satisfied for a given Romanian text.

IMPORTANT CONTEXT:
- The text is in Romanian and may contain political discourse
- Romanian political commentary often uses strong language that is culturally acceptable
- Distinguish between legitimate criticism and genuine toxicity
- Consider cultural and linguistic nuances specific to Romania

For each criterion type:
- ESSENTIAL (E): Check if the toxicity indicator is PRESENT in the text
- IMPORTANT (I): Check if the contextual factor is properly CONSIDERED
- PITFALL (P): Check if the classification error is AVOIDED (satisfied = error avoided)

Respond with ONLY a JSON object containing:
- "satisfied": true/false (whether the criterion is met)
- "reasoning": brief explanation in English (1-2 sentences)"""

EXPLICIT_EVAL_USER = """**Text (Romanian):**
{text}

**Criterion to evaluate:**
ID: {rubric_id}
Title: {title}
Category: {category}
Description: {description}

Evaluate whether this criterion is satisfied. For Pitfall criteria, "satisfied" means the error was AVOIDED.

Respond with JSON:"""


# =============================================================================
# IMPLICIT AGGREGATION PROMPTS
# =============================================================================
# These prompts pass all rubrics to the LLM for holistic evaluation.
# The LLM provides a single 1-10 Likert score considering all criteria.
# =============================================================================

IMPLICIT_EVAL_SYSTEM = """You are an expert evaluator for toxicity detection in Romanian political discourse.

Your task is to provide a holistic toxicity assessment score (1-10) for a Romanian text, considering all provided evaluation criteria.

SCORING GUIDE:
- 1-2: Clearly toxic content with multiple severe violations (hate speech, threats, harassment)
- 3-4: Toxic content with clear violations but less severe
- 5-6: Borderline content - may contain concerning elements but context-dependent
- 7-8: Generally acceptable content with minor concerns or strong but legitimate criticism
- 9-10: Non-toxic content that properly avoids all pitfalls

IMPORTANT CONTEXT FOR ROMANIAN DISCOURSE:
- Romanian political commentary often uses passionate, direct language
- Strong criticism of politicians is culturally acceptable and not inherently toxic
- Satire and irony are common in Romanian political discourse
- Consider cultural expressions that may seem harsh but are normal in Romanian context

You must weigh:
- Essential criteria (E): Critical toxicity indicators - heavily impact score if present
- Important criteria (I): Contextual factors - moderate impact on score
- Pitfall criteria (P): Errors to avoid - penalize score if these mistakes are made

Respond with a JSON object: {{"rating": <1-10>, "explanation": "<brief reasoning>"}}"""

IMPLICIT_EVAL_USER = """**Text (Romanian):**
{text}

**Evaluation Criteria (TORCH-RaR Framework):**

ESSENTIAL CRITERIA (must detect if present):
{essential_rubrics}

IMPORTANT CRITERIA (contextual factors):
{important_rubrics}

PITFALL CRITERIA (errors to avoid):
{pitfall_rubrics}

Provide your holistic toxicity assessment (1-10) considering ALL criteria above.
Respond with JSON:"""


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
        # Evaluate each criterion in parallel
        semaphore = asyncio.Semaphore(self.settings.max_concurrent_requests)

        async def eval_rubric(rubric: RubricItem) -> RubricEvaluation:
            async with semaphore:
                return await self._evaluate_single_rubric(text, rubric)

        tasks = [eval_rubric(r) for r in rubrics]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, handling failed evaluations gracefully
        evaluations: list[RubricEvaluation] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Rubric evaluation {i} failed: {result}")
                # Create a failed evaluation with score 0
                evaluations.append(RubricEvaluation(
                    rubric=rubrics[i],
                    rubric_id=rubrics[i].rubric_id,
                    satisfied=False,
                    score=0.0,
                    reasoning=f"Evaluation failed: {result}",
                ))
            else:
                evaluations.append(result)

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

        This implements the per-criterion evaluation for explicit aggregation.
        Each criterion (E1-E4, I1-I4, P1-P3) is evaluated independently.

        For Pitfall criteria (P1-P3):
        - satisfied=True means the error was AVOIDED (good)
        - satisfied=False means the error was MADE (bad, will be penalized)

        Args:
            text: The Romanian text to evaluate.
            rubric: The rubric criterion to check.

        Returns:
            RubricEvaluation with satisfaction status and reasoning.
        """
        messages = [
            {"role": "system", "content": EXPLICIT_EVAL_SYSTEM},
            {
                "role": "user",
                "content": EXPLICIT_EVAL_USER.format(
                    text=text,
                    rubric_id=rubric.rubric_id,
                    title=rubric.title,
                    category=rubric.category.value,
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
            satisfied = result.get("satisfied", False)

            return RubricEvaluation(
                rubric=rubric,
                rubric_id=rubric.rubric_id,
                satisfied=satisfied,
                score=1.0 if satisfied else 0.0,
                reasoning=result.get("reasoning"),
            )

        except Exception as e:
            logger.error(f"Failed to evaluate rubric '{rubric.rubric_id} - {rubric.title}': {e}")
            return RubricEvaluation(
                rubric=rubric,
                rubric_id=rubric.rubric_id,
                satisfied=False,
                score=0.0,
                reasoning=f"Evaluation failed: {e}",
            )

    def _parse_explicit_response(self, response: str) -> dict[str, Any]:
        """Parse the explicit evaluation response."""
        try:
            return extract_json_from_response(response, expected_type="object")
        except JSONParseError:
            # Fall back to boolean extraction from text
            satisfied = extract_boolean_from_response(response)
            return {"satisfied": satisfied, "reasoning": response}

    async def calculate_implicit_reward(
        self,
        text: str,
        rubrics: list[RubricItem],
    ) -> tuple[float, str]:
        """Calculate reward using implicit aggregation (Equation 2 from RaR paper).

        All rubric criteria are passed to the LLM-as-judge, organized by category,
        which produces a single scalar reward (1-10 Likert scale).

        This approach lets the LLM holistically consider all criteria rather than
        evaluating each independently, potentially capturing nuanced interactions.

        Args:
            text: The Romanian text to evaluate.
            rubrics: List of rubric criteria (E1-E4, I1-I4, P1-P3).

        Returns:
            Tuple of (normalized reward [0-1], raw response).
        """
        # Format rubrics by category for clearer prompt structure
        essential_rubrics = "\n".join(
            f"- [{r.rubric_id}] {r.title}: {r.description}"
            for r in rubrics
            if r.category == RubricCategory.ESSENTIAL
        )
        important_rubrics = "\n".join(
            f"- [{r.rubric_id}] {r.title}: {r.description}"
            for r in rubrics
            if r.category == RubricCategory.IMPORTANT
        )
        pitfall_rubrics = "\n".join(
            f"- [{r.rubric_id}] {r.title}: {r.description}"
            for r in rubrics
            if r.category == RubricCategory.PITFALL
        )

        messages = [
            {"role": "system", "content": IMPLICIT_EVAL_SYSTEM},
            {
                "role": "user",
                "content": IMPLICIT_EVAL_USER.format(
                    text=text,
                    essential_rubrics=essential_rubrics or "None specified",
                    important_rubrics=important_rubrics or "None specified",
                    pitfall_rubrics=pitfall_rubrics or "None specified",
                ),
            },
        ]

        try:
            response = await self.llm_client.complete(
                messages=messages,
                model_type="judge",
                temperature=0.1,
                max_tokens=512,
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
        return extract_rating_from_response(response, min_val=1, max_val=10)

    async def calculate_reward(
        self,
        text: str,
        rubrics: list[RubricItem],
        method: str = "both",
    ) -> RewardResult:
        """Calculate rewards for a text using specified method.

        This is the main entry point for reward calculation, supporting both
        explicit aggregation (per-criterion evaluation) and implicit aggregation
        (holistic LLM judge evaluation).

        Args:
            text: The Romanian text to evaluate.
            rubrics: List of rubric criteria (E1-E4, I1-I4, P1-P3).
            method: "explicit", "implicit", or "both".

        Returns:
            RewardResult with calculated rewards and category breakdown.

        Raises:
            ValidationError: If text is empty or rubrics are missing.
            RewardCalculationError: If reward calculation fails.
        """
        # Input validation
        if not text or not text.strip():
            raise ValidationError("Text cannot be empty for reward calculation")
        if not rubrics:
            raise ValidationError("At least one rubric is required for reward calculation")
        if method not in ("explicit", "implicit", "both"):
            raise ValidationError(f"Invalid method '{method}'. Must be 'explicit', 'implicit', or 'both'")

        explicit_reward = 0.0
        implicit_reward = None
        evaluations = []
        raw_response = None
        category_scores = None

        try:
            if method in ("explicit", "both"):
                explicit_reward, evaluations = await self.calculate_explicit_reward(
                    text, rubrics
                )
                # Calculate per-category scores for interpretability
                category_scores = self._calculate_category_scores(evaluations)

            if method in ("implicit", "both"):
                implicit_reward, raw_response = await self.calculate_implicit_reward(
                    text, rubrics
                )
        except ValidationError:
            raise
        except Exception as e:
            raise RewardCalculationError(f"Reward calculation failed: {e}") from e

        return RewardResult(
            explicit_reward=explicit_reward,
            implicit_reward=implicit_reward,
            evaluations=evaluations,
            raw_response=raw_response,
            category_scores=category_scores,
        )

    def _calculate_category_scores(
        self, evaluations: list[RubricEvaluation]
    ) -> dict[str, float]:
        """Calculate per-category satisfaction scores.

        This provides interpretability by showing how well each category
        of criteria was satisfied.

        Args:
            evaluations: List of RubricEvaluation results.

        Returns:
            Dictionary with category names as keys and satisfaction ratios as values.
        """
        category_counts: dict[str, list[bool]] = {
            "Essential": [],
            "Important": [],
            "Pitfall": [],
        }

        for eval_result in evaluations:
            category = eval_result.rubric.category.value
            if category in category_counts:
                category_counts[category].append(eval_result.satisfied)

        scores = {}
        for category, results in category_counts.items():
            if results:
                scores[category] = sum(results) / len(results)
            else:
                scores[category] = 0.0

        return scores

    async def calculate_rewards_batch(
        self,
        texts: list[str],
        rubrics_list: list[list[RubricItem]],
        method: str = "both",
    ) -> list[Optional[RewardResult]]:
        """Calculate rewards for multiple texts.

        Args:
            texts: List of texts to evaluate.
            rubrics_list: List of rubric lists, one per text.
            method: "explicit", "implicit", or "both".

        Returns:
            List of RewardResult objects. Failed calculations return None.
        """
        semaphore = asyncio.Semaphore(self.settings.max_concurrent_requests)

        async def limited_calc(text: str, rubrics: list[RubricItem]) -> RewardResult:
            async with semaphore:
                return await self.calculate_reward(text, rubrics, method)

        tasks = [limited_calc(t, r) for t, r in zip(texts, rubrics_list)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, converting exceptions to None
        processed: list[Optional[RewardResult]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch reward calculation {i} failed: {result}")
                processed.append(None)
            else:
                processed.append(result)

        return processed
