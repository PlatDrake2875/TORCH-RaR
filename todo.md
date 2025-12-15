# Goal

This is a research project about augumenting datasets such as https://huggingface.co/datasets/olimpia20/toxicity-dataset-ro-master using the ./Rubrics as Rewards Reinforcement Learning Beyond Verifiable Domains (1).pdf method. 

We want to use LiteLLM (https://docs.litellm.ai/docs/proxy/deploy) for inference and OpenRouter for models. We also want to leave space for a dockerized vLLM instance (read https://docs.vllm.ai/en/stable/deployment/docker/) in place of OpenRouter. Follow best practices and use uv (instead of pip)