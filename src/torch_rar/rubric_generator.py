"""Rubric generation module following the RaR (Rubrics as Rewards) method.

This module implements the rubric generation strategy from:
- "Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains" (Scale AI)
- TORCH-RaR framework for Romanian toxicity detection

IMPORTANT: These rubrics evaluate MODEL PREDICTIONS, not text content directly.
The RaR methodology uses rubrics to score how well a model's prediction
aligns with the ground truth label during RL training.

The rubrics follow a hierarchical structure (per guide Section 5.1):
- Essential (E1-E4): Critical classification factors
  - E1: correct_label_assignment, weight=1.0
  - E2: personal_attack_detection, weight=0.95
  - E3: threat_incitement_detection, weight=0.90
  - E4: group_hatred_detection, weight=0.90
- Important (I1-I4): Contextual quality factors
  - I1: contextual_appropriateness, weight=0.70
  - I2: emotional_intensity_recognition, weight=0.65
  - I3: sarcasm_implicit_toxicity, weight=0.60
  - I4: political_figure_recognition, weight=0.60
- Pitfall (P1-P3): Classification errors (penalties)
  - P1: false_positive_criticism, weight=-0.60
  - P2: false_negative_implicit, weight=-0.65
  - P3: context_free_classification, weight=-0.50

These rubrics are specifically designed for Romanian political discourse,
accounting for cultural expressions, political rhetoric, and regional context.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from torch_rar.config import RubricWeights, Settings
from torch_rar.exceptions import JSONParseError, RubricGenerationError, ValidationError
from torch_rar.json_utils import extract_json_from_response
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
    """A single rubric criterion following the RaR methodology.

    Each rubric item represents an evaluation criterion that can be:
    - Essential (E1-E4): Must be evaluated, critical for toxicity detection
    - Important (I1-I4): Should be evaluated, provides context
    - Pitfall (P1-P3): Common mistakes to avoid (negative weight = penalty)

    IMPORTANT: These rubrics evaluate MODEL PREDICTIONS, not text content directly.
    The RaR methodology uses rubrics to score how well a model's prediction
    aligns with the ground truth label, considering various quality factors.

    Attributes:
        rubric_id: Unique identifier (e.g., "E1", "I2", "P3")
        title: Short descriptive name (2-5 words)
        description: Full evaluation criterion description
        weight: Numerical weight for aggregation (negative for pitfalls)
        category: RubricCategory enum value
        evaluation_method: Method used to evaluate this criterion
        patterns: Optional list of Romanian patterns for pattern matching
        trigger_condition: For pitfalls, the condition that triggers the penalty
    """

    title: str
    description: str
    weight: float
    category: RubricCategory
    rubric_id: str = ""
    evaluation_method: str = "llm_judge"
    patterns: list[str] = field(default_factory=list)
    trigger_condition: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RubricItem":
        """Create RubricItem from dictionary.

        Args:
            data: Dictionary with keys: title, description, weight,
                  optionally rubric_id, category, evaluation_method, patterns

        Returns:
            RubricItem instance
        """
        desc = data.get("description", "")
        category_str = data.get("category", "")

        # Parse category from explicit field or description prefix
        category = RubricCategory.IMPORTANT  # default

        if category_str:
            try:
                category = RubricCategory(category_str)
            except ValueError:
                pass
        else:
            for cat in RubricCategory:
                if desc.lower().startswith(f"{cat.value.lower()} criteria"):
                    category = cat
                    break

        return cls(
            rubric_id=data.get("rubric_id", data.get("id", "")),
            title=data.get("title", ""),
            description=desc,
            weight=float(data.get("weight", 1.0)),
            category=category,
            evaluation_method=data.get("evaluation_method", "llm_judge"),
            patterns=data.get("patterns", []),
            trigger_condition=data.get("trigger_condition", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "rubric_id": self.rubric_id,
            "title": self.title,
            "description": self.description,
            "weight": self.weight,
            "category": self.category.value,
            "evaluation_method": self.evaluation_method,
        }
        if self.patterns:
            result["patterns"] = self.patterns
        if self.trigger_condition:
            result["trigger_condition"] = self.trigger_condition
        return result


# =============================================================================
# ROMANIAN-SPECIFIC RUBRIC PROMPTS
# =============================================================================
# These prompts are designed for evaluating toxicity in Romanian political
# discourse, following the RaR methodology. They account for:
# - Romanian language expressions and idioms
# - Political rhetoric common in Romanian discourse
# - Cultural context and regional expressions
# - The distinction between criticism and genuine toxicity
# =============================================================================

TOXICITY_RUBRIC_SYSTEM_PROMPT = """You are an expert rubric writer specializing in toxicity detection for Romanian political discourse and social media content.

Your task is to generate evaluation criteria ("rubrics") that capture the nuances of:
- Romanian language expressions and idioms that may appear offensive out of context
- Political rhetoric common in Romanian public discourse
- Cultural and regional expressions specific to Romania
- The distinction between legitimate political criticism and genuine toxicity

IMPORTANT CONTEXT:
The dataset contains Romanian political discourse where:
- Strong political opinions are common and not inherently toxic
- Criticism of politicians or parties is legitimate discourse
- Some expressions have cultural significance that must be considered
- Satire and irony are frequently used in Romanian political commentary

Rubric categories follow the RaR (Rubrics as Rewards) methodology:
- ESSENTIAL (E1-E4): Critical toxicity indicators, weight=1.0
- IMPORTANT (I1-I4): Contextual factors for assessment, weight=0.7
- PITFALL (P1-P3): Common misclassification errors to avoid, weight=-0.9

Each rubric must be self-contained and explicitly state what to evaluate."""

TOXICITY_RUBRIC_USER_TEMPLATE = """Generate evaluation rubrics for the following Romanian text that may contain toxic content.

**Text (Romanian):**
{text}

**Task:** Create {min_items}-{max_items} rubric items following the TORCH-RaR framework for Romanian toxicity evaluation.

**Required Structure:**
Your rubrics MUST include items from each category:
- At least 3-4 Essential criteria (E) - critical toxicity indicators
- At least 3-4 Important criteria (I) - contextual factors
- At least 2-3 Pitfall criteria (P) - errors to avoid

**Output Format:** JSON array with objects containing:
1. `rubric_id`: Category code (E1, E2, I1, I2, P1, etc.)
2. `title`: 2-5 words describing the criterion
3. `description`: Full evaluation criterion in Romanian context
4. `weight`: 1.0 for Essential, 0.7 for Important, -0.9 for Pitfall
5. `category`: "Essential", "Important", or "Pitfall"

**Example for Romanian context:**
```json
[
  {{"rubric_id": "E1", "title": "Discurs de Ură Explicit", "description": "Identifică limbaj explicit de ură care vizează grupuri protejate (etnie, religie, gen, orientare sexuală) - include insulte rasiale, antisemite sau xenofobe în limba română.", "weight": 1.0, "category": "Essential"}},
  {{"rubric_id": "I2", "title": "Retorică Politică vs Toxicitate", "description": "Distinge între opinii politice puternice sau critici legitime ale politicienilor și atacuri cu adevărat toxice - critica politică în România poate fi dură fără a fi toxică.", "weight": 0.7, "category": "Important"}},
  {{"rubric_id": "P1", "title": "Supraclasificare a Criticii", "description": "Evită etichetarea criticii legitime, dezacordului sau opiniilor negative ca fiind toxice - românii folosesc frecvent limbaj direct în discursul politic.", "weight": -0.9, "category": "Pitfall"}}
]
```

Generate the rubrics now as a JSON array:"""


# =============================================================================
# TORCH-RaR PREDEFINED RUBRICS (E1-E4, I1-I4, P1-P3)
# =============================================================================
# These rubrics follow the methodology from:
# 1. "Rubrics as Rewards" paper (Scale AI) - hierarchical rubric structure
# 2. TORCH-RaR guide Section 5.1 - Romanian political discourse specifics
#
# IMPORTANT: These rubrics evaluate MODEL PREDICTIONS, not text content.
# Each rubric scores how well the model's prediction aligns with ground truth.
#
# The weights follow the RaR paper's explicit aggregation formula:
#   r(x, ŷ) = Σⱼ(wⱼ · cⱼ(x, ŷ)) / Σⱼ|wⱼ|
#
# Where:
#   - x is the input text
#   - ŷ is the model's prediction
#   - wⱼ is the weight for criterion j (guide Section 5.1 weights)
#   - cⱼ(x, ŷ) ∈ {0, 1} indicates if criterion j is satisfied
#   - Pitfall weights are negative (penalty when the error occurs)
# =============================================================================


def get_torch_rar_rubrics() -> list[RubricItem]:
    """Get the complete set of TORCH-RaR rubrics for Romanian toxicity detection.

    IMPORTANT: These rubrics evaluate MODEL PREDICTIONS, not text content directly.
    Following the RaR (Rubrics as Rewards) methodology, each rubric scores how well
    the model's prediction aligns with the ground truth label.

    The rubrics are organized into three categories (per guide Section 5.1):

    ESSENTIAL (E1-E4) - Weights: 1.0, 0.95, 0.90, 0.90
    --------------------------------------------------
    Critical factors for correct classification. These evaluate whether the model
    correctly identified fundamental toxicity indicators.

    IMPORTANT (I1-I4) - Weights: 0.70, 0.65, 0.60, 0.60
    ---------------------------------------------------
    Contextual factors that improve classification quality. These evaluate whether
    the model considered nuanced aspects of the text.

    PITFALL (P1-P3) - Weights: -0.60, -0.65, -0.50 (penalties)
    ----------------------------------------------------------
    Common classification errors. The negative weight penalizes the reward when
    the model makes these mistakes.

    Explicit aggregation formula (from RaR paper):
        r(x, ŷ) = Σⱼ(wⱼ · cⱼ(x, ŷ)) / Σⱼ|wⱼ|

    Returns:
        List of 11 RubricItem objects following the TORCH-RaR framework
    """
    rubrics = [
        # =====================================================================
        # ESSENTIAL CRITERIA (E1-E4)
        # Critical factors for correct toxicity classification
        # These evaluate whether the MODEL correctly identified key indicators
        # =====================================================================
        RubricItem(
            rubric_id="E1",
            title="Correct Label Assignment",
            description=(
                "Modelul a prezis corect eticheta de toxicitate (0 sau 1) "
                "comparativ cu ground truth-ul. Aceasta este verificarea "
                "fundamentală: predicția modelului se potrivește cu eticheta "
                "corectă din dataset? Evaluare: comparație directă între "
                "prediction și ground_truth_label."
            ),
            weight=RubricWeights.E1_CORRECT_LABEL,
            category=RubricCategory.ESSENTIAL,
            evaluation_method="direct_comparison",
        ),
        RubricItem(
            rubric_id="E2",
            title="Personal Attack Detection",
            description=(
                "Modelul a identificat corect prezența sau absența atacurilor "
                "personale în text. Dacă textul conține insulte personale "
                "(pattern-uri: prost*, idiot*, nenorocit*, dobitoc*, etc.) și "
                "ground truth=TOXIC, modelul trebuie să prezică TOXIC. Dacă "
                "aceste pattern-uri lipsesc și ground truth=NON-TOXIC, modelul "
                "trebuie să prezică NON-TOXIC."
            ),
            weight=RubricWeights.E2_PERSONAL_ATTACK,
            category=RubricCategory.ESSENTIAL,
            evaluation_method="pattern_matching_with_llm_judge",
            patterns=["prost*", "idiot*", "nenorocit*", "dobitoc*", "cretin*",
                      "imbecil*", "tâmpit*", "bou", "vită", "animal"],
        ),
        RubricItem(
            rubric_id="E3",
            title="Threat/Incitement Detection",
            description=(
                "Modelul a identificat corect prezența sau absența amenințărilor "
                "și incitării la violență. Pattern-uri cheie: 'vor plăti', "
                "'la pușcărie', 'trebuie oprit*', 'merită bătaie', 'să moară'. "
                "Când aceste indicatori există și ground truth=TOXIC, modelul "
                "trebuie să prezică corect. Evaluează alinierea predicției cu "
                "prezența reală a amenințărilor."
            ),
            weight=RubricWeights.E3_THREAT_DETECTION,
            category=RubricCategory.ESSENTIAL,
            evaluation_method="pattern_matching_with_semantic_analysis",
            patterns=["vor plăti", "la pușcărie", "trebuie oprit*", "merită bătaie",
                      "să moară", "la zid", "spânzurat*", "împușcat*"],
        ),
        RubricItem(
            rubric_id="E4",
            title="Group Hatred Detection",
            description=(
                "Modelul a identificat corect prezența sau absența urii de grup "
                "(dehumanizare, limbaj despre 'infestare'). Pattern-uri: 'ciumă', "
                "'infecție', 'gângaci', 'șobolani', 'paraziți', 'jigodii'. "
                "Evaluează dacă predicția modelului se aliniază cu prezența "
                "reală a limbajului de ură de grup conform ground truth-ului."
            ),
            weight=RubricWeights.E4_GROUP_HATRED,
            category=RubricCategory.ESSENTIAL,
            evaluation_method="pattern_matching_with_contextual_analysis",
            patterns=["ciumă", "infecție", "gângaci", "șobolani", "paraziți",
                      "jigodii", "lighioane", "dobitoace", "lepre"],
        ),

        # =====================================================================
        # IMPORTANT CRITERIA (I1-I4)
        # Contextual factors that improve classification quality
        # These evaluate whether the MODEL considered nuanced aspects
        # =====================================================================
        RubricItem(
            rubric_id="I1",
            title="Contextual Appropriateness",
            description=(
                "Modelul a luat în considerare contextul când a făcut predicția. "
                "Un model bun nu clasifică doar pe baza cuvintelor cheie, ci "
                "înțelege contextul în care apar. Evaluează dacă predicția "
                "modelului demonstrează înțelegere contextuală - nu doar "
                "pattern matching superficial."
            ),
            weight=RubricWeights.I1_CONTEXTUAL,
            category=RubricCategory.IMPORTANT,
            evaluation_method="llm_judge_with_context",
        ),
        RubricItem(
            rubric_id="I2",
            title="Emotional Intensity Recognition",
            description=(
                "Modelul a recunoscut corect intensitatea emoțională a textului "
                "și a clasificat în consecință. Text cu intensitate emoțională "
                "ridicată nu este automat toxic - critica pasională poate fi "
                "legitimă. Evaluează dacă modelul a distins între emoție "
                "puternică și toxicitate reală."
            ),
            weight=RubricWeights.I2_EMOTIONAL,
            category=RubricCategory.IMPORTANT,
            evaluation_method="sentiment_analysis_with_llm_judge",
        ),
        RubricItem(
            rubric_id="I3",
            title="Sarcasm/Implicit Toxicity Handling",
            description=(
                "Modelul a gestionat corect sarcasmul și toxicitatea implicită. "
                "Textul sarcastic poate părea inofensiv la suprafață dar să fie "
                "toxic, sau poate părea dur dar să fie glumă. Evaluează dacă "
                "predicția modelului se aliniază cu ground truth-ul pentru "
                "cazurile de sarcasm și ironie."
            ),
            weight=RubricWeights.I3_SARCASM,
            category=RubricCategory.IMPORTANT,
            evaluation_method="llm_judge_specialized",
        ),
        RubricItem(
            rubric_id="I4",
            title="Political Figure Recognition",
            description=(
                "Modelul a recunoscut corect când textul vizează figuri politice "
                "și a clasificat în consecință. Politicieni relevanți: Iohannis, "
                "Ciolacu, Lasconi, Simion, Georgescu, Antonescu, Geoană. "
                "Partide: PSD, AUR, USR, PNL. Critica politicienilor nu este "
                "automat toxică - evaluează dacă modelul a făcut această distincție."
            ),
            weight=RubricWeights.I4_POLITICAL,
            category=RubricCategory.IMPORTANT,
            evaluation_method="ner_with_targeting_analysis",
            patterns=["Iohannis", "Ciolacu", "Lasconi", "Simion", "Georgescu",
                      "Antonescu", "Geoană", "PSD", "AUR", "USR", "PNL",
                      "Ciucă", "Băsescu", "Dragnea"],
        ),

        # =====================================================================
        # PITFALL CRITERIA (P1-P3) - Negative weights (penalties)
        # Common classification errors that should be avoided
        # These PENALIZE the reward when the model makes these mistakes
        # =====================================================================
        RubricItem(
            rubric_id="P1",
            title="False Positive on Criticism",
            description=(
                "PENALIZARE: Modelul a clasificat greșit critica legitimă ca "
                "fiind toxică. Condiție de declanșare: prediction=TOXIC dar "
                "ground_truth=NON-TOXIC și textul conține critică politică "
                "legitimă, dezacord, sau opinii negative fără atacuri personale. "
                "Această eroare este frecventă când modelul reacționează la "
                "tonul negativ fără a evalua conținutul real."
            ),
            weight=RubricWeights.P1_FALSE_POSITIVE,
            category=RubricCategory.PITFALL,
            evaluation_method="false_positive_detection",
            trigger_condition="prediction=TOXIC AND ground_truth=NON-TOXIC AND text_is_legitimate_criticism",
        ),
        RubricItem(
            rubric_id="P2",
            title="False Negative on Implicit Toxicity",
            description=(
                "PENALIZARE: Modelul a ratat toxicitatea implicită sau subtilă. "
                "Condiție de declanșare: prediction=NON-TOXIC dar ground_truth=TOXIC "
                "și textul conține toxicitate mascată (dehumanizare subtilă, "
                "insinuări, dog whistles). Această eroare apare când modelul "
                "se bazează doar pe cuvinte cheie explicite."
            ),
            weight=RubricWeights.P2_FALSE_NEGATIVE,
            category=RubricCategory.PITFALL,
            evaluation_method="false_negative_detection",
            trigger_condition="prediction=NON-TOXIC AND ground_truth=TOXIC AND text_has_implicit_toxicity",
        ),
        RubricItem(
            rubric_id="P3",
            title="Context-Free Classification",
            description=(
                "PENALIZARE: Modelul a clasificat bazându-se doar pe cuvinte "
                "cheie, ignorând complet contextul. Condiție de declanșare: "
                "predicție incorectă care ar fi fost corectă dacă modelul ar fi "
                "considerat contextul (ex: citat, discuție despre toxicitate, "
                "negație). Această eroare indică o înțelegere superficială."
            ),
            weight=RubricWeights.P3_CONTEXT_FREE,
            category=RubricCategory.PITFALL,
            evaluation_method="context_analysis",
            trigger_condition="incorrect_prediction AND context_would_change_classification",
        ),
    ]

    return rubrics


def get_rubric_by_id(rubric_id: str) -> Optional[RubricItem]:
    """Get a specific rubric by its ID.

    Args:
        rubric_id: The rubric identifier (e.g., "E1", "I2", "P3")

    Returns:
        RubricItem if found, None otherwise
    """
    rubrics = get_torch_rar_rubrics()
    for rubric in rubrics:
        if rubric.rubric_id == rubric_id:
            return rubric
    return None


def get_rubrics_by_category(category: RubricCategory) -> list[RubricItem]:
    """Get all rubrics of a specific category.

    Args:
        category: RubricCategory (ESSENTIAL, IMPORTANT, or PITFALL)

    Returns:
        List of RubricItems in that category
    """
    rubrics = get_torch_rar_rubrics()
    return [r for r in rubrics if r.category == category]


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

        Raises:
            ValidationError: If text is empty or exceeds limits.
            RubricGenerationError: If rubric generation fails.
        """
        # Input validation
        if not text or not text.strip():
            raise ValidationError("Text cannot be empty for rubric generation")
        if len(text) > 50000:
            raise ValidationError("Text exceeds maximum length of 50000 characters")

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

        except JSONParseError as e:
            logger.error(f"Failed to parse rubrics response: {e}")
            raise RubricGenerationError(f"Failed to parse rubrics: {e}") from e
        except Exception as e:
            logger.error(f"Failed to generate rubrics: {e}")
            raise RubricGenerationError(f"Failed to generate rubrics: {e}") from e

    def _parse_rubrics(self, response: str) -> list[RubricItem]:
        """Parse rubrics from LLM response.

        Args:
            response: Raw LLM response string.

        Returns:
            List of parsed RubricItem objects.

        Raises:
            JSONParseError: If JSON cannot be extracted or parsed.
        """
        data = extract_json_from_response(response, expected_type="array")
        return [RubricItem.from_dict(item) for item in data]

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
            List of rubric lists, one per input text. Failed generations return empty lists.
        """
        refs = reference_answers or [None] * len(texts)
        semaphore = asyncio.Semaphore(self.settings.max_concurrent_requests)

        async def limited_generate(text: str, ref: Optional[str]) -> list[RubricItem]:
            async with semaphore:
                return await self.generate_rubrics(text, ref)

        tasks = [limited_generate(t, r) for t, r in zip(texts, refs)]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, converting exceptions to empty lists
        processed: list[list[RubricItem]] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch rubric generation {i} failed: {result}")
                processed.append([])
            else:
                processed.append(result)

        return processed

    def get_predefined_rubrics(self) -> list[RubricItem]:
        """Get predefined TORCH-RaR rubrics for Romanian toxicity detection.

        These rubrics implement the E1-E4, I1-I4, P1-P3 framework from the
        TORCH-RaR methodology, specifically designed for Romanian political
        discourse following the RaR paper's approach.

        The RaR paper distinguishes between:
        - RaR-INSTANCE: Rubrics generated per-instance (more accurate, costly)
        - RaR-PREDEFINED: Static rubrics applied to all samples (faster)

        This method returns RaR-PREDEFINED rubrics tailored for Romanian context.

        Weight scheme (from RaR paper):
        - Essential (E): weight=1.0 (critical indicators)
        - Important (I): weight=0.7 (contextual factors)
        - Pitfall (P): weight=-0.9 (penalty for misclassification)

        Returns:
            List of 11 RubricItem objects (4 Essential + 4 Important + 3 Pitfall)
        """
        return get_torch_rar_rubrics()
