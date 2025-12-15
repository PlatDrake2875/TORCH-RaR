"""Rubric generation module following the RaR (Rubrics as Rewards) method."""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from torch_rar.config import Settings
from torch_rar.llm_client import LLMClient

logger = logging.getLogger(__name__)


class RubricCategory(str, Enum):
    """Categories for rubric criteria."""

    ESSENTIAL = "Essential"
    IMPORTANT = "Important"
    OPTIONAL = "Optional"
    PITFALL = "Pitfall"


@dataclass
class RubricItem:
    """A single rubric criterion."""

    title: str
    description: str
    weight: float
    category: RubricCategory

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RubricItem":
        """Create RubricItem from dictionary."""
        # Parse category from description prefix
        desc = data.get("description", "")
        category = RubricCategory.IMPORTANT  # default

        for cat in RubricCategory:
            if desc.lower().startswith(f"{cat.value.lower()} criteria"):
                category = cat
                break

        return cls(
            title=data.get("title", ""),
            description=desc,
            weight=float(data.get("weight", 1.0)),
            category=category,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "description": self.description,
            "weight": self.weight,
            "category": self.category.value,
        }


TOXICITY_RUBRIC_SYSTEM_PROMPT = """You are an expert rubric writer for toxicity detection and content moderation. Your job is to generate a self-contained set of evaluation criteria ("rubrics") for judging responses related to potentially toxic or harmful content.

Rubrics should cover aspects such as:
- Toxicity indicators (hate speech, harassment, threats, discrimination)
- Language severity and context
- Target identification (individuals, groups, protected characteristics)
- Intent assessment (malicious vs. frustrated vs. misunderstood)
- Cultural and contextual sensitivity
- Potential for harm

Each rubric item must be fully self-contained so that non-expert readers need not consult any external information."""

TOXICITY_RUBRIC_USER_TEMPLATE = """Generate evaluation rubrics for the following text that may contain toxic content.

**Text to evaluate:**
{text}

**Task:** Create {min_items}-{max_items} rubric items to evaluate whether this text is toxic and to what degree. Each rubric item should help assess different aspects of potential toxicity.

**Output Format:** Provide a JSON array of rubric objects. Each object must contain exactly three keys:
1. `title`: 2-4 words describing the criterion
2. `description`: One sentence beginning with its category prefix that explicitly states what to look for
   - Essential Criteria: Critical toxicity indicators that strongly suggest harmful content
   - Important Criteria: Significant factors in assessing toxicity level
   - Optional Criteria: Additional context that may inform the assessment
   - Pitfall Criteria: Common misclassifications to avoid (use "Does not..." format)
3. `weight`: For Essential/Important/Optional, use 1-5 (5 = most important); for Pitfall, use -1 or -2

**Example rubric items:**
```json
[
  {{"title": "Hate Speech Presence", "description": "Essential Criteria: Identifies explicit hate speech targeting protected characteristics such as race, religion, gender, or sexual orientation.", "weight": 5}},
  {{"title": "Threat Assessment", "description": "Essential Criteria: Detects direct or implied threats of violence or harm to individuals or groups.", "weight": 5}},
  {{"title": "Dehumanizing Language", "description": "Important Criteria: Identifies language that dehumanizes or degrades individuals or groups.", "weight": 4}},
  {{"title": "Context Consideration", "description": "Important Criteria: Considers whether potentially offensive language is used in a clearly sarcastic, educational, or quoted context.", "weight": 3}},
  {{"title": "Avoid Overclassification", "description": "Pitfall Criteria: Does not classify mere disagreement, criticism, or negative sentiment as toxic without evidence of harmful intent.", "weight": -1}}
]
```

Generate the rubrics now as a JSON array:"""


class RubricGenerator:
    """Generate instance-specific rubrics for toxicity evaluation following RaR method."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        """Initialize the rubric generator.

        Args:
            settings: Configuration settings.
            llm_client: LLM client for API calls. If None, creates new one.
        """
        self.settings = settings or Settings()
        self.llm_client = llm_client or LLMClient(self.settings)

    async def generate_rubrics(
        self,
        text: str,
        reference_answer: Optional[str] = None,
        min_items: Optional[int] = None,
        max_items: Optional[int] = None,
    ) -> list[RubricItem]:
        """Generate rubrics for evaluating a text sample.

        Args:
            text: The text to generate rubrics for.
            reference_answer: Optional reference answer for expert grounding.
            min_items: Minimum number of rubric items.
            max_items: Maximum number of rubric items.

        Returns:
            List of RubricItem objects.
        """
        min_items = min_items or self.settings.min_rubric_items
        max_items = max_items or self.settings.max_rubric_items

        user_prompt = TOXICITY_RUBRIC_USER_TEMPLATE.format(
            text=text,
            min_items=min_items,
            max_items=max_items,
        )

        if reference_answer:
            user_prompt += f"\n\n**Reference Assessment:**\n{reference_answer}"

        messages = [
            {"role": "system", "content": TOXICITY_RUBRIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await self.llm_client.complete(
                messages=messages,
                model_type="rubric",
                temperature=0.3,
                max_tokens=4096,
            )

            rubrics = self._parse_rubrics(response)
            logger.info(f"Generated {len(rubrics)} rubrics for text sample")
            return rubrics

        except Exception as e:
            logger.error(f"Failed to generate rubrics: {e}")
            raise

    def _parse_rubrics(self, response: str) -> list[RubricItem]:
        """Parse rubrics from LLM response.

        Args:
            response: Raw LLM response string.

        Returns:
            List of parsed RubricItem objects.
        """
        # Extract JSON from response
        text = response.strip()

        # Handle markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end]
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end]

        # Find JSON array
        if "[" in text:
            start = text.find("[")
            end = text.rfind("]") + 1
            text = text[start:end]

        try:
            data = json.loads(text)
            return [RubricItem.from_dict(item) for item in data]
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse rubrics JSON: {e}")
            logger.debug(f"Raw response: {response}")
            return []

    async def generate_rubrics_batch(
        self,
        texts: list[str],
        reference_answers: Optional[list[str]] = None,
    ) -> list[list[RubricItem]]:
        """Generate rubrics for multiple texts.

        Args:
            texts: List of texts to generate rubrics for.
            reference_answers: Optional list of reference answers.

        Returns:
            List of rubric lists, one per input text.
        """
        import asyncio

        refs = reference_answers or [None] * len(texts)
        semaphore = asyncio.Semaphore(self.settings.max_concurrent_requests)

        async def limited_generate(text: str, ref: Optional[str]) -> list[RubricItem]:
            async with semaphore:
                return await self.generate_rubrics(text, ref)

        tasks = [limited_generate(t, r) for t, r in zip(texts, refs)]
        return await asyncio.gather(*tasks, return_exceptions=False)

    def get_predefined_rubrics(self) -> list[RubricItem]:
        """Get predefined static rubrics for toxicity detection.

        These are generic rubrics that can be applied to any text sample,
        following the RaR-PREDEFINED approach from the paper.

        Returns:
            List of predefined RubricItem objects.
        """
        predefined = [
            RubricItem(
                title="Hate Speech Detection",
                description="Essential Criteria: The response correctly identifies explicit hate speech targeting protected characteristics.",
                weight=5.0,
                category=RubricCategory.ESSENTIAL,
            ),
            RubricItem(
                title="Threat Identification",
                description="Essential Criteria: The response accurately detects direct or implied threats of violence.",
                weight=5.0,
                category=RubricCategory.ESSENTIAL,
            ),
            RubricItem(
                title="Harassment Recognition",
                description="Important Criteria: The response identifies patterns of harassment or targeted abuse.",
                weight=4.0,
                category=RubricCategory.IMPORTANT,
            ),
            RubricItem(
                title="Discrimination Assessment",
                description="Important Criteria: The response recognizes discriminatory language or bias.",
                weight=4.0,
                category=RubricCategory.IMPORTANT,
            ),
            RubricItem(
                title="Profanity Context",
                description="Important Criteria: The response evaluates profanity in context rather than as standalone toxicity.",
                weight=3.0,
                category=RubricCategory.IMPORTANT,
            ),
            RubricItem(
                title="Cultural Sensitivity",
                description="Optional Criteria: The response considers cultural and linguistic context in assessment.",
                weight=2.0,
                category=RubricCategory.OPTIONAL,
            ),
            RubricItem(
                title="Avoid False Positives",
                description="Pitfall Criteria: Does not classify benign disagreement or criticism as toxic.",
                weight=-1.0,
                category=RubricCategory.PITFALL,
            ),
        ]
        return predefined
