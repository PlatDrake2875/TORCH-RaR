# TORCH-RaR Implementation Guide

This document provides comprehensive research and implementation guidance for the TORCH-RaR project features outlined in `todo.md`. Use this as a reference for implementing the remaining features.

---

## Table of Contents

1. [Project Context](#project-context)
2. [Rubrics as Rewards (RaR) Methodology](#rubrics-as-rewards-rar-methodology)
3. [CRITICAL: Generalized Prompt Template System](#critical-generalized-prompt-template-system)
4. [HIGH PRIORITY: Cached Rubric Generation](#high-priority-cached-rubric-generation)
5. [HIGH PRIORITY: Progress Bar](#high-priority-progress-bar)
6. [HIGH PRIORITY: Logs Generation](#high-priority-logs-generation)
7. [Implementation Priorities](#implementation-priorities)
8. [References](#references)

---

## Project Context

### Current Architecture

```
src/torch_rar/
├── __init__.py           # Package exports
├── config.py             # Settings loaded from settings.yaml
├── llm_client.py         # LiteLLM wrapper for OpenRouter/vLLM
├── data_loader.py        # HuggingFace dataset loading
├── rubric_generator.py   # Instance-specific rubric generation
├── reward_calculator.py  # Explicit/implicit reward aggregation
└── pipeline.py           # Main augmentation pipeline
```

### Key Files and Line References

| File | Purpose | Key Sections |
|------|---------|--------------|
| `rubric_generator.py` | Rubric generation | Lines 144-205 (prompts), 230-449 (predefined) |
| `pipeline.py` | Main data flow | Lines 49-103 (sample), 105-213 (run) |
| `reward_calculator.py` | Reward calculation | Lines 93-179 (prompts), 207-272 (explicit), 347-412 (implicit) |
| `config.py` | Settings management | Lines 13-48 (weights), 110-288 (Settings class) |
| `llm_client.py` | LLM integration | Lines 69-119 (complete methods) |

### Current Prompt Locations

All prompts are currently hardcoded as string constants:

1. **rubric_generator.py**:
   - `TOXICITY_RUBRIC_SYSTEM_PROMPT` (lines 154-174)
   - `TOXICITY_RUBRIC_USER_TEMPLATE` (lines 176-205)

2. **reward_calculator.py**:
   - `EXPLICIT_EVAL_SYSTEM` (lines 100-117)
   - `EXPLICIT_EVAL_USER` (lines 119-130)
   - `IMPLICIT_EVAL_SYSTEM` (lines 140-162)
   - `IMPLICIT_EVAL_USER` (lines 164-179)

---

## Rubrics as Rewards (RaR) Methodology

### Paper Reference

- **Title**: Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains
- **Authors**: Anisha Gunjal et al. (Scale AI)
- **arXiv**: [2507.17746](https://arxiv.org/abs/2507.17746)
- **Local Copy**: `docs/Rubrics_as_Rewards_paper.pdf`

### Core Concept

RaR extends RLVR (Reinforcement Learning with Verifiable Rewards) to domains without clear correctness signals by using structured, checklist-style rubrics as interpretable reward signals for on-policy training with GRPO.

### Problem Formulation

Let `x` denote an input prompt and `ŷ ~ πθ(· | x)` be a sampled response. Each prompt is associated with a set of k rubric items `{(wj, cj)}` where:
- `wj ∈ ℝ` = weight of criterion j
- `cj: (x, ŷ) → {0, 1}` = binary correctness function

### Rubric Categories and Weights

| Category | Weight | Purpose |
|----------|--------|---------|
| **Essential** | 1.0 | Critical facts/safety checks - if missing, response is invalid |
| **Important** | 0.7 | Key reasoning, completeness, clarity - strongly affects quality |
| **Optional** | 0.3 | Helpful style or extra depth - nice to have |
| **Pitfall** | 0.9 (penalty) | Common mistakes to avoid - negative contribution if not satisfied |

### Reward Aggregation Strategies

#### 1. Explicit Aggregation

Each criterion evaluated independently using LLM-as-judge, normalized weighted sum:

```
r(x, ŷ) = Σ(wj · cj(x, ŷ)) / Σ(wj)
```

**Advantages**: More control, interpretable per-criterion scores
**Disadvantages**: Fixed weights can be brittle, requires tuning

#### 2. Implicit Aggregation

All rubric criteria passed to LLM judge for holistic scoring:

```
r_implicit(x, ŷ) = fφ(x, ŷ, {dj})
```

Where `fφ` is an LLM-based judge that produces a 1-10 Likert score.

**Advantages**: No weight tuning needed, captures criterion interactions
**Disadvantages**: Less interpretable, depends on judge quality

### Four Desiderata for Rubric Generation

1. **Grounded in Expert Guidance**: Rubrics should reflect domain expertise, capturing essential facts, reasoning steps, and conclusions. Use reference answers as proxies for expert supervision.

2. **Comprehensive Coverage**: Rubrics should span multiple dimensions:
   - Factual accuracy
   - Logical coherence
   - Completeness
   - Style and safety
   - Negative criteria (pitfalls) for common errors

3. **Criterion Importance**: Some dimensions are more critical than others. Factual correctness must outweigh secondary aspects like stylistic clarity.

4. **Self-Contained Evaluation**: Each rubric item should be independently actionable, allowing assessment in isolation without external context.

### Rubric Generation Best Practices (from paper)

- Generate 7-20 self-contained items per prompt
- Each item needs:
  - `title`: 2-4 words
  - `description`: One sentence with category prefix
  - `weight`: 1-5 for positive, -1 or -2 for pitfall
- Use stronger LLMs (GPT-4o, o3-mini) with reference answers for generation
- Synthetic rubrics with reference guidance outperform those without

---

## CRITICAL: Generalized Prompt Template System

### Requirements from todo.md

```
[] Generalised method for generating rubric system/user prompts. The variable parts should be:
- System prompt: tasks, important context
- User prompt: tasks, examples
```

### Recommended Architecture

#### Option 1: Jinja2-Based Template System (Recommended)

```python
# src/torch_rar/prompt_templates.py
from jinja2 import Environment, FileSystemLoader, select_autoescape
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

@dataclass
class PromptTemplate:
    """Represents a system/user prompt pair."""
    system: str
    user: str

    def format(self, **kwargs) -> tuple[str, str]:
        return self.system.format(**kwargs), self.user.format(**kwargs)

class PromptTemplateRegistry:
    """
    Registry for loading and managing prompt templates.

    Supports:
    - Domain-specific templates (toxicity, medical, science)
    - Variable substitution via Jinja2
    - Template inheritance
    - Default fallbacks
    """

    def __init__(self, templates_dir: str = "prompts/"):
        self.templates_dir = Path(templates_dir)
        self.env = Environment(
            loader=FileSystemLoader(templates_dir),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        self._cache: Dict[str, PromptTemplate] = {}

    def get_rubric_prompts(
        self,
        domain: str,
        tasks: List[str],
        context: Optional[str] = None,
        examples: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> PromptTemplate:
        """
        Load domain-specific rubric generation prompts.

        Args:
            domain: Domain name (e.g., 'toxicity', 'medical', 'science')
            tasks: List of task descriptions for the system prompt
            context: Optional important context for the domain
            examples: Optional list of example rubrics for few-shot
            **kwargs: Additional template variables

        Returns:
            PromptTemplate with rendered system and user prompts
        """
        cache_key = f"rubric_{domain}"

        try:
            system_tpl = self.env.get_template(f"{domain}/rubric_system.jinja2")
            user_tpl = self.env.get_template(f"{domain}/rubric_user.jinja2")
        except Exception:
            # Fall back to default templates
            system_tpl = self.env.get_template("default/rubric_system.jinja2")
            user_tpl = self.env.get_template("default/rubric_user.jinja2")

        template_vars = {
            "domain": domain,
            "tasks": tasks,
            "context": context,
            "examples": examples or [],
            **kwargs
        }

        return PromptTemplate(
            system=system_tpl.render(**template_vars),
            user=user_tpl.render(**template_vars)
        )

    def get_evaluation_prompts(
        self,
        method: str,  # "explicit" or "implicit"
        domain: str,
        **kwargs
    ) -> PromptTemplate:
        """Load evaluation prompts for reward calculation."""
        try:
            system_tpl = self.env.get_template(f"{domain}/{method}_eval_system.jinja2")
            user_tpl = self.env.get_template(f"{domain}/{method}_eval_user.jinja2")
        except Exception:
            system_tpl = self.env.get_template(f"default/{method}_eval_system.jinja2")
            user_tpl = self.env.get_template(f"default/{method}_eval_user.jinja2")

        return PromptTemplate(
            system=system_tpl.render(domain=domain, **kwargs),
            user=user_tpl.render(domain=domain, **kwargs)
        )
```

#### Template Directory Structure

```
prompts/
├── default/
│   ├── rubric_system.jinja2
│   ├── rubric_user.jinja2
│   ├── explicit_eval_system.jinja2
│   ├── explicit_eval_user.jinja2
│   ├── implicit_eval_system.jinja2
│   └── implicit_eval_user.jinja2
├── toxicity/
│   ├── rubric_system.jinja2
│   ├── rubric_user.jinja2
│   └── ... (evaluation templates)
├── medical/
│   └── ... (domain-specific templates)
└── science/
    └── ... (domain-specific templates)
```

#### Example Template: `prompts/default/rubric_system.jinja2`

```jinja2
You are an expert rubric writer for {{ domain }} evaluation tasks.

{% if context %}
## Important Context
{{ context }}
{% endif %}

## Your Tasks
{% for task in tasks %}
- {{ task }}
{% endfor %}

## Rubric Categories
Generate rubrics in these categories with appropriate weights:
- **Essential** (weight 5): Critical factors - if missing, response is invalid
- **Important** (weight 3-4): Key reasoning and completeness factors
- **Optional** (weight 1-2): Nice-to-have style or extra depth
- **Pitfall** (weight -1 to -2): Common mistakes to avoid

## Requirements
- Generate 7-20 self-contained rubric items
- Each item must have: title (2-4 words), description (one sentence), weight
- Description must start with category prefix (e.g., "Essential Criteria: ...")
- Items must be independently evaluable without external context
```

#### Example Template: `prompts/default/rubric_user.jinja2`

```jinja2
Generate evaluation rubrics for the following {{ domain }} text.

## Input Text
{{ text }}

{% if examples %}
## Example Rubrics
{% for example in examples %}
### Example {{ loop.index }}
Text: {{ example.text }}
Rubrics:
```json
{{ example.rubrics | tojson(indent=2) }}
```
{% endfor %}
{% endif %}

## Instructions
- Generate between {{ min_items | default(7) }} and {{ max_items | default(20) }} rubric items
- Use weights: Essential=5, Important=3-4, Optional=1-2, Pitfall=-1 to -2
- Output as a JSON array of objects with keys: title, description, weight

## Output Format
```json
[
  {"title": "...", "description": "Essential Criteria: ...", "weight": 5},
  {"title": "...", "description": "Important Criteria: ...", "weight": 4},
  {"title": "...", "description": "Pitfall Criteria: Does not ...", "weight": -1}
]
```
```

#### Configuration Integration

```yaml
# settings.yaml
prompt_templates:
  directory: "prompts/"
  default_domain: "toxicity"
  domains:
    toxicity:
      tasks:
        - "Evaluate Romanian political discourse for toxicity"
        - "Identify personal attacks, threats, and group hatred"
      context: "Focus on Romanian language and cultural context"
    medical:
      tasks:
        - "Evaluate medical response accuracy and safety"
        - "Check for appropriate disclaimers"
      context: "Medical domain requires safety-first evaluation"
    science:
      tasks:
        - "Evaluate scientific reasoning and accuracy"
        - "Check for proper methodology references"
```

#### Integration with RubricGenerator

```python
# src/torch_rar/rubric_generator.py (modified)
from .prompt_templates import PromptTemplateRegistry

class RubricGenerator:
    def __init__(self, llm_client, settings):
        self.llm_client = llm_client
        self.settings = settings
        self.template_registry = PromptTemplateRegistry(
            settings.prompt_templates.directory
        )

    async def generate_rubrics(
        self,
        text: str,
        domain: str = None,
        examples: list = None
    ) -> list[RubricItem]:
        domain = domain or self.settings.prompt_templates.default_domain
        domain_config = self.settings.prompt_templates.domains.get(domain, {})

        prompts = self.template_registry.get_rubric_prompts(
            domain=domain,
            tasks=domain_config.get("tasks", []),
            context=domain_config.get("context"),
            examples=examples,
            text=text,
            min_items=self.settings.min_rubric_items,
            max_items=self.settings.max_rubric_items
        )

        response = await self.llm_client.complete(
            system_prompt=prompts.system,
            user_prompt=prompts.user,
            temperature=0.3,
            max_tokens=4096
        )

        return self._parse_rubrics(response)
```

---

## HIGH PRIORITY: Cached Rubric Generation

### Current State

No caching implemented - every API call goes through fresh. This is costly and slow for repeated/similar texts.

### Recommended Implementation: DiskCache

```python
# src/torch_rar/cache.py
import hashlib
import json
from pathlib import Path
from typing import Optional, Callable, Any
from dataclasses import dataclass, asdict
from diskcache import Cache
import logging

logger = logging.getLogger(__name__)

@dataclass
class CacheConfig:
    """Configuration for rubric caching."""
    enabled: bool = True
    directory: str = ".cache/rubrics"
    ttl_seconds: int = 2592000  # 30 days
    size_limit_gb: float = 1.0

class RubricCache:
    """
    Persistent cache for generated rubrics.

    Features:
    - Deterministic hashing based on input text + model
    - Automatic TTL expiration
    - Thread-safe disk storage
    - Cache statistics tracking
    """

    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache = Cache(
            config.directory,
            size_limit=int(config.size_limit_gb * 1024 * 1024 * 1024)
        )
        self._stats = {"hits": 0, "misses": 0}

    def _create_cache_key(self, text: str, model: str, domain: str) -> str:
        """Create deterministic hash for cache lookup."""
        content = json.dumps({
            "text": text.strip().lower(),  # Normalize
            "model": model,
            "domain": domain
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    def get(self, text: str, model: str, domain: str) -> Optional[list]:
        """Retrieve cached rubrics if available."""
        if not self.config.enabled:
            return None

        key = self._create_cache_key(text, model, domain)
        result = self.cache.get(key)

        if result is not None:
            self._stats["hits"] += 1
            logger.debug(f"Cache hit for key {key[:8]}...")
        else:
            self._stats["misses"] += 1

        return result

    def set(self, text: str, model: str, domain: str, rubrics: list) -> None:
        """Store rubrics in cache."""
        if not self.config.enabled:
            return

        key = self._create_cache_key(text, model, domain)
        self.cache.set(key, rubrics, expire=self.config.ttl_seconds)
        logger.debug(f"Cached rubrics for key {key[:8]}...")

    async def get_or_generate(
        self,
        text: str,
        model: str,
        domain: str,
        generator_fn: Callable[[], Any]
    ) -> list:
        """
        Get from cache or generate and cache.

        Args:
            text: Input text
            model: Model name used for generation
            domain: Domain (e.g., 'toxicity')
            generator_fn: Async function to generate rubrics if not cached

        Returns:
            List of rubrics (from cache or freshly generated)
        """
        cached = self.get(text, model, domain)
        if cached is not None:
            return cached

        # Generate fresh rubrics
        rubrics = await generator_fn()
        self.set(text, model, domain, rubrics)
        return rubrics

    def get_stats(self) -> dict:
        """Return cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0
        return {
            **self._stats,
            "total_requests": total,
            "hit_rate": f"{hit_rate:.2%}",
            "cache_size_mb": self.cache.volume() / (1024 * 1024)
        }

    def clear(self) -> None:
        """Clear all cached rubrics."""
        self.cache.clear()
        self._stats = {"hits": 0, "misses": 0}
```

### Alternative: LiteLLM Native Caching

Since your project already uses LiteLLM, you can enable its built-in caching:

```yaml
# settings.yaml
litellm_settings:
  cache: true
  cache_params:
    type: "disk"  # or "redis" for distributed
    disk_cache_dir: ".cache/litellm"
    ttl: 2592000  # 30 days in seconds
```

```python
# src/torch_rar/llm_client.py (modified)
import litellm

class LLMClient:
    def __init__(self, settings):
        if settings.litellm_settings.cache:
            litellm.cache = litellm.Cache(
                type=settings.litellm_settings.cache_params.type,
                disk_cache_dir=settings.litellm_settings.cache_params.disk_cache_dir,
                ttl=settings.litellm_settings.cache_params.ttl
            )
```

### Integration with RubricGenerator

```python
# src/torch_rar/rubric_generator.py (modified)
class RubricGenerator:
    def __init__(self, llm_client, settings, cache: RubricCache = None):
        self.llm_client = llm_client
        self.settings = settings
        self.cache = cache or RubricCache(settings.cache)

    async def generate_rubrics(self, text: str, domain: str = "toxicity") -> list:
        async def _generate():
            # Actual LLM call
            response = await self.llm_client.complete(...)
            return self._parse_rubrics(response)

        return await self.cache.get_or_generate(
            text=text,
            model=self.settings.rubric_generator_model,
            domain=domain,
            generator_fn=_generate
        )
```

### Configuration

```yaml
# settings.yaml
cache:
  enabled: true
  directory: ".cache/rubrics"
  ttl_seconds: 2592000  # 30 days
  size_limit_gb: 1.0
```

---

## HIGH PRIORITY: Progress Bar

### Current State

Basic tqdm usage in `pipeline.py:150` for batch progress.

### Enhanced Implementation with tqdm.asyncio and rich

```python
# src/torch_rar/progress.py
from typing import Optional, Iterable, AsyncIterable
from contextlib import contextmanager
import asyncio

from tqdm.asyncio import tqdm_asyncio
from tqdm.auto import tqdm
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn
)
from rich.console import Console

console = Console()

class ProgressTracker:
    """
    Unified progress tracking for sync and async operations.

    Features:
    - Automatic detection of sync/async context
    - Rich progress bars with detailed stats
    - Nested progress support
    - Statistics collection
    """

    def __init__(self, disable: bool = False):
        self.disable = disable
        self._stats = {
            "processed": 0,
            "failed": 0,
            "cached": 0
        }

    @contextmanager
    def track_pipeline(self, total: int, description: str = "Processing"):
        """Context manager for pipeline-level progress."""
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
            refresh_per_second=2
        ) as progress:
            task = progress.add_task(description, total=total)

            class ProgressContext:
                def __init__(self, progress, task, stats):
                    self._progress = progress
                    self._task = task
                    self._stats = stats

                def advance(self, n: int = 1, cached: bool = False):
                    self._progress.advance(self._task, n)
                    self._stats["processed"] += n
                    if cached:
                        self._stats["cached"] += n

                def fail(self, n: int = 1):
                    self._stats["failed"] += n

                def update_description(self, desc: str):
                    self._progress.update(self._task, description=desc)

            yield ProgressContext(progress, task, self._stats)

    async def gather_with_progress(
        self,
        coroutines: list,
        description: str = "Processing"
    ) -> list:
        """
        Async gather with progress bar.

        Uses tqdm_asyncio for efficient async progress tracking.
        """
        if self.disable:
            return await asyncio.gather(*coroutines, return_exceptions=True)

        return await tqdm_asyncio.gather(
            *coroutines,
            desc=description,
            leave=False
        )

    def iterate_with_progress(
        self,
        iterable: Iterable,
        total: Optional[int] = None,
        description: str = "Processing"
    ):
        """Sync iteration with progress bar."""
        if self.disable:
            yield from iterable
            return

        yield from tqdm(
            iterable,
            total=total,
            desc=description,
            leave=False
        )

    def get_stats(self) -> dict:
        """Return progress statistics."""
        return {
            **self._stats,
            "success_rate": (
                f"{(self._stats['processed'] - self._stats['failed']) / self._stats['processed']:.2%}"
                if self._stats['processed'] > 0 else "N/A"
            ),
            "cache_rate": (
                f"{self._stats['cached'] / self._stats['processed']:.2%}"
                if self._stats['processed'] > 0 else "N/A"
            )
        }
```

### Integration with Pipeline

```python
# src/torch_rar/pipeline.py (modified)
from .progress import ProgressTracker

class AugmentationPipeline:
    def __init__(self, settings):
        self.settings = settings
        self.progress = ProgressTracker(disable=not settings.show_progress)

    async def run(self, limit: Optional[int] = None) -> list[AugmentedSample]:
        samples = list(self.data_loader.iter_samples(limit=limit))
        results = []

        with self.progress.track_pipeline(
            total=len(samples),
            description="[cyan]Augmenting dataset"
        ) as tracker:
            for batch in self._batch(samples, self.settings.batch_size):
                # Process batch concurrently
                batch_results = await self.progress.gather_with_progress(
                    [self._process_sample(s) for s in batch],
                    description="Batch processing"
                )

                for result in batch_results:
                    if result is not None:
                        results.append(result)
                        tracker.advance(1, cached=result.from_cache)
                    else:
                        tracker.fail(1)
                        tracker.advance(1)

        # Log final stats
        stats = self.progress.get_stats()
        logger.info(f"Pipeline complete: {stats}")

        return results
```

---

## HIGH PRIORITY: Logs Generation

### Recommended: Loguru with Structured Output

```python
# src/torch_rar/logging_config.py
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger
from typing import Optional

def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    json_logs: bool = True,
    rotation: str = "10 MB",
    retention: str = "7 days"
) -> None:
    """
    Configure logging for the TORCH-RaR pipeline.

    Features:
    - Colored console output for development
    - JSON-formatted file logs for production/analysis
    - Automatic log rotation and retention
    - Context binding for structured logging

    Args:
        log_dir: Directory for log files
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR)
        json_logs: Whether to output JSON to files
        rotation: When to rotate log files
        retention: How long to keep old logs
    """
    # Create log directory
    log_path = Path(log_dir)
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
        level=log_level,
        colorize=True
    )

    # File handler - JSON format for parsing
    timestamp = datetime.now().strftime("%Y%m%d")

    if json_logs:
        logger.add(
            log_path / f"pipeline_{timestamp}.json",
            format="{message}",
            level=log_level,
            serialize=True,  # JSON output
            rotation=rotation,
            retention=retention,
            compression="gz"
        )
    else:
        logger.add(
            log_path / f"pipeline_{timestamp}.log",
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                "{name}:{function}:{line} | {message}"
            ),
            level=log_level,
            rotation=rotation,
            retention=retention,
            compression="gz"
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
        retention="30 days"
    )

    logger.info("Logging initialized", log_dir=str(log_path), level=log_level)


def get_logger(name: str):
    """
    Get a contextualized logger for a module.

    Usage:
        logger = get_logger(__name__)
        logger.info("Processing sample", sample_id=123)
    """
    return logger.bind(module=name)
```

### Usage Examples

```python
# src/torch_rar/rubric_generator.py
from .logging_config import get_logger

logger = get_logger(__name__)

class RubricGenerator:
    async def generate_rubrics(self, text: str) -> list:
        log = logger.bind(text_length=len(text), domain=self.domain)
        log.info("Starting rubric generation")

        try:
            rubrics = await self._call_llm(text)
            log.bind(rubric_count=len(rubrics)).info("Rubrics generated successfully")
            return rubrics
        except Exception as e:
            log.exception("Failed to generate rubrics")
            raise


# src/torch_rar/pipeline.py
from .logging_config import get_logger

logger = get_logger(__name__)

class AugmentationPipeline:
    async def process_sample(self, sample: ToxicitySample) -> Optional[AugmentedSample]:
        log = logger.bind(sample_id=sample.id, text_preview=sample.text[:50])

        log.debug("Processing sample")

        try:
            rubrics = await self.rubric_generator.generate_rubrics(sample.text)
            rewards = await self.reward_calculator.calculate(sample, rubrics)

            log.bind(
                rubric_count=len(rubrics),
                explicit_reward=rewards.explicit,
                implicit_reward=rewards.implicit
            ).info("Sample processed successfully")

            return AugmentedSample(sample=sample, rubrics=rubrics, rewards=rewards)

        except Exception as e:
            log.exception("Failed to process sample")
            return None
```

### Configuration

```yaml
# settings.yaml
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  directory: "logs"
  json_format: true
  rotation: "10 MB"
  retention: "7 days"
```

### Log Analysis

With JSON logs, you can analyze pipeline performance:

```python
import json
from pathlib import Path

def analyze_logs(log_file: str):
    """Analyze pipeline logs for statistics."""
    stats = {
        "total_samples": 0,
        "successful": 0,
        "failed": 0,
        "avg_rubrics": [],
        "avg_rewards": []
    }

    with open(log_file) as f:
        for line in f:
            entry = json.loads(line)
            record = entry.get("record", {})

            if "sample_id" in record.get("extra", {}):
                stats["total_samples"] += 1

                if record.get("level", {}).get("name") == "ERROR":
                    stats["failed"] += 1
                elif "rubric_count" in record.get("extra", {}):
                    stats["successful"] += 1
                    stats["avg_rubrics"].append(record["extra"]["rubric_count"])
                    if "explicit_reward" in record["extra"]:
                        stats["avg_rewards"].append(record["extra"]["explicit_reward"])

    return {
        "total_samples": stats["total_samples"],
        "successful": stats["successful"],
        "failed": stats["failed"],
        "success_rate": f"{stats['successful']/stats['total_samples']:.2%}" if stats["total_samples"] > 0 else "N/A",
        "avg_rubrics_per_sample": sum(stats["avg_rubrics"]) / len(stats["avg_rubrics"]) if stats["avg_rubrics"] else 0,
        "avg_explicit_reward": sum(stats["avg_rewards"]) / len(stats["avg_rewards"]) if stats["avg_rewards"] else 0
    }
```

---

## Implementation Priorities

| Feature | Complexity | Impact | Dependencies | Recommended Order |
|---------|------------|--------|--------------|-------------------|
| **Generalized Prompt Templates** | Medium | High | Jinja2 | 1 |
| **Cached Rubric Generation** | Low | High | diskcache | 2 |
| **Structured Logging** | Low | Medium | loguru | 3 |
| **Progress Bar Enhancement** | Low | Medium | rich, tqdm | 4 |

### Suggested Implementation Steps

1. **Week 1: Prompt Templates**
   - Create `prompts/` directory structure
   - Implement `PromptTemplateRegistry` class
   - Migrate existing prompts to Jinja2 templates
   - Update `RubricGenerator` and `RewardCalculator`
   - Add domain configuration to `settings.yaml`

2. **Week 2: Caching**
   - Implement `RubricCache` class
   - Integrate with `RubricGenerator`
   - Add cache configuration to settings
   - Test cache hit rates

3. **Week 3: Logging & Progress**
   - Set up Loguru configuration
   - Add structured logging throughout pipeline
   - Enhance progress bars with rich
   - Add statistics collection

---

## References

### Papers and Documentation

1. **Rubrics as Rewards Paper**
   - arXiv: https://arxiv.org/abs/2507.17746
   - Scale AI: https://scale.com/research/rubrics_as_rewards
   - Local: `docs/Rubrics_as_Rewards_paper.pdf`

2. **Released Datasets**
   - RaR-Medicine: https://huggingface.co/datasets/anisha2102/RaR-Medicine
   - RaR-Science: https://huggingface.co/datasets/anisha2102/RaR-Science

### Libraries

3. **Prompt Templates**
   - Jinja2: https://jinja.palletsprojects.com/
   - Prompt Poet (Character.AI): https://github.com/character-ai/prompt-poet
   - LangChain PromptTemplate: https://python.langchain.com/api_reference/core/prompts/

4. **Caching**
   - DiskCache: https://grantjenks.com/docs/diskcache/
   - LiteLLM Caching: https://docs.litellm.ai/docs/proxy/caching
   - GPTCache: https://github.com/zilliztech/GPTCache

5. **Logging**
   - Loguru: https://github.com/Delgan/loguru
   - Structlog: https://www.structlog.org/
   - Better Stack Guide: https://betterstack.com/community/guides/logging/best-python-logging-libraries/

6. **Progress Bars**
   - tqdm: https://github.com/tqdm/tqdm
   - Rich: https://github.com/Textualize/rich
   - tqdm asyncio: https://www.dataleadsfuture.com/using-tqdm-with-asyncio-in-python/

---

## Appendix: Quick Reference

### Rubric Category Weights (from paper)

```python
RUBRIC_WEIGHTS = {
    "Essential": 1.0,
    "Important": 0.7,
    "Optional": 0.3,
    "Pitfall": 0.9  # Applied as penalty
}
```

### Explicit Aggregation Formula

```
r(x, ŷ) = Σ(wj · cj(x, ŷ)) / Σ(wj)
```

### Required Dependencies

```toml
# pyproject.toml additions
[project.dependencies]
jinja2 = ">=3.1.0"
diskcache = ">=5.6.0"
loguru = ">=0.7.0"
rich = ">=13.0.0"
tqdm = ">=4.66.0"
```
