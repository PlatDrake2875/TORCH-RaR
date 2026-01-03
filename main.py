#!/usr/bin/env python3
"""TORCH-RaR: Rubrics as Rewards for Toxicity Dataset Augmentation.

This CLI tool augments toxicity datasets using the RaR (Rubrics as Rewards)
method from the paper "Rubrics as Rewards: Reinforcement Learning Beyond
Verifiable Domains".
"""

import argparse
import asyncio
import sys

from loguru import logger


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="TORCH-RaR: Augment toxicity datasets using Rubrics as Rewards",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (10 samples, both reward methods)
  python main.py run --limit 10

  # Use predefined rubrics for faster processing
  python main.py run --limit 100 --predefined-rubrics

  # Use only implicit reward calculation
  python main.py run --limit 50 --reward-method implicit

  # Specify a custom config file
  python main.py --config custom_settings.yaml run --limit 20

Configuration:
  Settings are loaded from settings.yaml in the project root.
  API keys can use environment variable substitution: ${OPENROUTER_API_KEY}
        """,
    )

    # Global options
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to settings.yaml file (default: auto-detect)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run the augmentation pipeline")
    run_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of samples to process (default: all)",
    )
    run_parser.add_argument(
        "--predefined-rubrics",
        action="store_true",
        help="Use predefined static rubrics instead of generating them",
    )
    run_parser.add_argument(
        "--reward-method",
        choices=["explicit", "implicit", "both"],
        default="both",
        help="Reward calculation method (default: both)",
    )
    run_parser.add_argument(
        "--output-format",
        choices=["parquet", "json", "csv"],
        default="parquet",
        help="Output file format (default: parquet)",
    )
    run_parser.add_argument(
        "--text-column",
        type=str,
        default=None,
        help="Name of text column in dataset (auto-detected if not specified)",
    )
    run_parser.add_argument(
        "--label-column",
        type=str,
        default=None,
        help="Name of label column in dataset (auto-detected if not specified)",
    )
    run_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    # Test command
    test_parser = subparsers.add_parser("test", help="Test the configuration")
    test_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "run":
        return run_pipeline_command(args)
    elif args.command == "test":
        return test_config_command(args)

    return 0


def run_pipeline_command(args: argparse.Namespace) -> int:
    """Execute the pipeline run command."""
    try:
        from torch_rar.config import load_settings
        from torch_rar.logging_config import setup_logging
        from torch_rar.pipeline import AugmentationPipeline

        # Load settings from YAML
        settings = load_settings(args.config)

        # Setup logging with config
        setup_logging(settings.logging, verbose=args.verbose)

        logger.info("Starting TORCH-RaR pipeline...")
        logger.info(f"  Dataset: {settings.dataset_name}")
        logger.info(f"  Provider: {settings.llm_provider.value}")
        logger.info(f"  Rubric model: {settings.rubric_generator_model}")
        logger.info(f"  Judge model: {settings.judge_model}")

        pipeline = AugmentationPipeline(settings)

        samples, stats = asyncio.run(
            pipeline.run(
                limit=args.limit,
                use_predefined_rubrics=args.predefined_rubrics,
                reward_method=args.reward_method,
                output_format=args.output_format,
                text_column=args.text_column,
                label_column=args.label_column,
            )
        )

        logger.info("Pipeline completed successfully!")
        return 0

    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        logger.info("Create settings.yaml from the template or specify --config path")
        return 1
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


def test_config_command(args: argparse.Namespace) -> int:
    """Test the configuration and connectivity."""
    try:
        from torch_rar.config import load_settings
        from torch_rar.llm_client import LLMClient
        from torch_rar.logging_config import setup_logging

        # Load settings from YAML
        settings = load_settings(args.config)

        # Setup logging with config
        setup_logging(settings.logging, verbose=args.verbose)

        logger.info("Testing configuration...")
        logger.info(f"  LLM Provider: {settings.llm_provider.value}")
        logger.info(f"  Rubric model: {settings.rubric_generator_model}")
        logger.info(f"  Judge model: {settings.judge_model}")
        logger.info(f"  Dataset: {settings.dataset_name}")

        # Check API key
        api_key = settings.get_api_key()
        if api_key:
            logger.info(f"  API Key: {'*' * 8}...{api_key[-4:]}")
        else:
            logger.warning("  API Key: Not set!")

        # Test API connectivity
        logger.info("Testing LLM API connectivity...")
        client = LLMClient(settings)

        async def test_completion():
            response = await client.complete(
                messages=[{"role": "user", "content": "Say 'Hello' in one word."}],
                model_type="judge",
                max_tokens=10,
            )
            return response

        response = asyncio.run(test_completion())
        logger.info(f"  API test successful! Response: {response.strip()}")

        logger.info("All tests passed!")
        return 0

    except FileNotFoundError as e:
        logger.error(f"Configuration error: {e}")
        logger.info("Create settings.yaml from the template or specify --config path")
        return 1
    except Exception as e:
        logger.exception(f"Configuration test failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
