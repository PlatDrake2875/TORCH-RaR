"""Prompt template management using Jinja2 for multi-domain support.

This module provides a flexible prompt template system that:
- Supports domain-specific templates (toxicity, medical, science)
- Falls back to default templates when domain-specific ones are unavailable
- Uses Jinja2 for variable substitution and conditional content
- Integrates with the Settings configuration

Example usage:
    registry = PromptTemplateRegistry("prompts/")
    prompts = registry.get_rubric_prompts(
        domain="toxicity",
        tasks=["Evaluate Romanian political discourse for toxicity"],
        context="Focus on Romanian language and cultural context",
        text="Sample text here",
        min_items=7,
        max_items=20
    )
    system_prompt = prompts.system
    user_prompt = prompts.user
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from jinja2 import Environment, FileSystemLoader, TemplateNotFound

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Represents a system/user prompt pair for LLM interactions.

    Attributes:
        system: The system prompt content
        user: The user prompt content
    """

    system: str
    user: str


class PromptTemplateRegistry:
    """Registry for loading and managing prompt templates.

    Supports:
    - Domain-specific templates (toxicity, medical, science)
    - Variable substitution via Jinja2
    - Automatic fallback to default templates
    - Template caching for performance

    Template Directory Structure:
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
        │   └── rubric_user.jinja2
        ├── medical/
        │   └── ...
        └── science/
            └── ...
    """

    def __init__(self, templates_dir: str = "prompts/"):
        """Initialize the template registry.

        Args:
            templates_dir: Path to the templates directory (relative or absolute)
        """
        self.templates_dir = Path(templates_dir)

        # Ensure templates directory exists
        if not self.templates_dir.exists():
            logger.warning(f"Templates directory not found: {self.templates_dir}")
            self.templates_dir.mkdir(parents=True, exist_ok=True)

        self.env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=False,
        )

    def _load_template(self, domain: str, template_name: str) -> str:
        """Load a template with fallback to default domain.

        Args:
            domain: The domain name (e.g., 'toxicity', 'medical')
            template_name: The template file name (e.g., 'rubric_system.jinja2')

        Returns:
            Rendered template string

        Raises:
            TemplateNotFound: If neither domain nor default template exists
        """
        # Try domain-specific template first
        domain_path = f"{domain}/{template_name}"
        try:
            return self.env.get_template(domain_path)
        except TemplateNotFound:
            logger.debug(f"Domain template not found: {domain_path}, falling back to default")

        # Fall back to default template
        default_path = f"default/{template_name}"
        return self.env.get_template(default_path)

    def get_rubric_prompts(
        self,
        domain: str,
        tasks: Optional[list[str]] = None,
        context: Optional[str] = None,
        examples: Optional[list[dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> PromptTemplate:
        """Load domain-specific rubric generation prompts.

        Args:
            domain: Domain name (e.g., 'toxicity', 'medical', 'science')
            tasks: List of task descriptions for the system prompt
            context: Optional important context for the domain
            examples: Optional list of example rubrics for few-shot
            **kwargs: Additional template variables (text, min_items, max_items, etc.)

        Returns:
            PromptTemplate with rendered system and user prompts
        """
        system_tpl = self._load_template(domain, "rubric_system.jinja2")
        user_tpl = self._load_template(domain, "rubric_user.jinja2")

        template_vars = {
            "domain": domain,
            "tasks": tasks or [],
            "context": context,
            "examples": examples or [],
            **kwargs,
        }

        return PromptTemplate(
            system=system_tpl.render(**template_vars),
            user=user_tpl.render(**template_vars),
        )

    def get_evaluation_prompts(
        self,
        method: str,
        domain: str,
        **kwargs: Any,
    ) -> PromptTemplate:
        """Load evaluation prompts for reward calculation.

        Args:
            method: Evaluation method ('explicit' or 'implicit')
            domain: Domain name (e.g., 'toxicity', 'medical')
            **kwargs: Template variables (text, rubric details, etc.)

        Returns:
            PromptTemplate with rendered system and user prompts
        """
        system_tpl = self._load_template(domain, f"{method}_eval_system.jinja2")
        user_tpl = self._load_template(domain, f"{method}_eval_user.jinja2")

        return PromptTemplate(
            system=system_tpl.render(domain=domain, **kwargs),
            user=user_tpl.render(domain=domain, **kwargs),
        )

    def template_exists(self, domain: str, template_name: str) -> bool:
        """Check if a specific template exists.

        Args:
            domain: Domain name
            template_name: Template file name

        Returns:
            True if template exists in domain or default
        """
        domain_path = self.templates_dir / domain / template_name
        default_path = self.templates_dir / "default" / template_name
        return domain_path.exists() or default_path.exists()

    def list_domains(self) -> list[str]:
        """List all available domains with templates.

        Returns:
            List of domain names that have template directories
        """
        domains = []
        if self.templates_dir.exists():
            for path in self.templates_dir.iterdir():
                if path.is_dir() and path.name != "default":
                    domains.append(path.name)
        return sorted(domains)
