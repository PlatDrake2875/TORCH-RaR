# TORCH-RaR: Rubrics as Rewards for Romanian Toxicity Detection

**Technical Documentation - Project Implementation Report**

---

## Abstract

This document describes the implementation of the TORCH-RaR framework, which applies the Rubrics as Rewards (RaR) methodology to Romanian toxicity detection. The framework augments the `olimpia20/toxicity-dataset-ro-master` dataset with structured evaluation rubrics and reward signals, preparing it for Reinforcement Learning (RL) training using Group Relative Policy Optimization (GRPO).

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Methodology](#2-methodology)
3. [Implementation](#3-implementation)
4. [Rubric Specification](#4-rubric-specification)
5. [Pipeline Architecture](#5-pipeline-architecture)
6. [Results](#6-results)
7. [Future Work](#7-future-work)
8. [References](#8-references)

---

## 1. Introduction

### 1.1 Problem Statement

Traditional toxicity detection models are trained using binary labels (toxic/non-toxic), which provide limited feedback during training. The model learns *what* is toxic but not *why* it is toxic. This leads to:

- Poor generalization to nuanced cases
- High false positive rates on legitimate criticism
- Inability to distinguish between strong opinions and genuine toxicity

### 1.2 Proposed Solution

The Rubrics as Rewards (RaR) methodology addresses these limitations by:

1. Defining structured evaluation criteria (rubrics) that capture different aspects of toxicity
2. Using these rubrics to generate richer reward signals during training
3. Enabling models to learn the *reasoning* behind toxicity classifications

### 1.3 Project Scope

This implementation focuses on Romanian political discourse, adapting the RaR methodology to:

- Romanian language patterns and expressions
- Political rhetoric common in Romanian discourse
- Cultural context specific to Romania

---

## 2. Methodology

### 2.1 RaR Framework Overview

The RaR methodology, introduced by Scale AI, replaces simple binary rewards with rubric-based evaluation. The key insight is that structured feedback enables more effective reinforcement learning.

### 2.2 Reward Aggregation

Two reward aggregation strategies are implemented:

#### 2.2.1 Explicit Aggregation

The explicit reward is calculated using a weighted sum of rubric scores:

```
r(x, ŷ) = Σⱼ(wⱼ · cⱼ(x, ŷ)) / Σⱼ|wⱼ|
```

Where:
- `x` is the input text
- `ŷ` is the model's prediction
- `wⱼ` is the weight for criterion j
- `cⱼ(x, ŷ) ∈ {0, 1}` indicates if criterion j is satisfied

#### 2.2.2 Implicit Aggregation

The implicit reward is obtained through holistic LLM evaluation on a 1-10 Likert scale, normalized to [0, 1].

### 2.3 Rubric Categories

Following the RaR paper, rubrics are organized into three categories:

| Category | Purpose | Weight Range |
|----------|---------|--------------|
| Essential (E) | Critical classification factors | 0.90 - 1.00 |
| Important (I) | Contextual quality factors | 0.60 - 0.70 |
| Pitfall (P) | Classification errors to penalize | -0.50 - -0.65 |

---

## 3. Implementation

### 3.1 Project Structure

```
TORCH-RaR/
├── src/torch_rar/
│   ├── __init__.py           # Package exports
│   ├── config.py             # Settings management
│   ├── llm_client.py         # LLM API wrapper (LiteLLM)
│   ├── data_loader.py        # Dataset loading and saving
│   ├── rubric_generator.py   # Rubric definitions (E1-E4, I1-I4, P1-P3)
│   ├── reward_calculator.py  # Reward computation
│   └── pipeline.py           # Orchestration
├── main.py                   # CLI entry point
├── settings.yaml             # Configuration
├── rubrics.json              # Rubric documentation
└── output/
    ├── augmented_dataset.parquet  # Binary output
    └── augmented_dataset.json     # Readable output
```

### 3.2 Configuration

The system supports multiple LLM providers:

```yaml
# settings.yaml
llm_provider: openrouter          # Options: openrouter, vllm, litellm_proxy
judge_model: openrouter/openai/gpt-4o-mini
```

### 3.3 Dependencies

- **LiteLLM**: Unified LLM API interface
- **Datasets**: HuggingFace dataset loading
- **Pandas/PyArrow**: Data manipulation and storage
- **Pydantic**: Configuration validation

---

## 4. Rubric Specification

### 4.1 Essential Criteria (E1-E4)

These criteria identify fundamental toxicity indicators.

| ID | Name | Weight | Evaluation Method |
|----|------|--------|-------------------|
| E1 | Correct Label Assignment | 1.00 | direct_comparison |
| E2 | Personal Attack Detection | 0.95 | pattern_matching_with_llm_judge |
| E3 | Threat/Incitement Detection | 0.90 | pattern_matching_with_semantic_analysis |
| E4 | Group Hatred Detection | 0.90 | pattern_matching_with_contextual_analysis |

**E2 Patterns (Romanian):**
```
prost*, idiot*, nenorocit*, dobitoc*, cretin*, imbecil*, tâmpit*, bou, vită, animal
```

**E3 Patterns (Romanian):**
```
vor plăti, la pușcărie, trebuie oprit*, merită bătaie, să moară, la zid, spânzurat*, împușcat*
```

**E4 Patterns (Romanian):**
```
ciumă, infecție, gângaci, șobolani, paraziți, jigodii, lighioane, dobitoace, lepre
```

### 4.2 Important Criteria (I1-I4)

These criteria evaluate contextual aspects of classification quality.

| ID | Name | Weight | Evaluation Method |
|----|------|--------|-------------------|
| I1 | Contextual Appropriateness | 0.70 | llm_judge_with_context |
| I2 | Emotional Intensity Recognition | 0.65 | sentiment_analysis_with_llm_judge |
| I3 | Sarcasm/Implicit Toxicity Handling | 0.60 | llm_judge_specialized |
| I4 | Political Figure Recognition | 0.60 | ner_with_targeting_analysis |

**I4 Political Entities:**
- Politicians: Iohannis, Ciolacu, Lasconi, Simion, Georgescu, Antonescu, Geoană, Ciucă, Băsescu, Dragnea
- Parties: PSD, AUR, USR, PNL

### 4.3 Pitfall Criteria (P1-P3)

These criteria penalize common classification errors.

| ID | Name | Weight | Trigger Condition |
|----|------|--------|-------------------|
| P1 | False Positive on Criticism | -0.60 | prediction=TOXIC ∧ ground_truth=NON-TOXIC ∧ legitimate_criticism |
| P2 | False Negative on Implicit Toxicity | -0.65 | prediction=NON-TOXIC ∧ ground_truth=TOXIC ∧ implicit_toxicity |
| P3 | Context-Free Classification | -0.50 | incorrect_prediction ∧ context_ignored |

### 4.4 Weight Summary

```
Total Positive Weight:  6.35  (E1 + E2 + E3 + E4 + I1 + I2 + I3 + I4)
Total Negative Weight: -1.75  (P1 + P2 + P3)
Total Absolute Weight:  8.10  (normalization denominator)
```

---

## 5. Pipeline Architecture

### 5.1 Data Flow

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  HuggingFace    │────▶│  Data Loader    │────▶│  Text Samples   │
│  Dataset        │     │                 │     │  (ToxicitySample)│
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Augmented      │◀────│  Reward         │◀────│  Rubric         │
│  Dataset        │     │  Calculator     │     │  Generator      │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                │
                                ▼
                        ┌─────────────────┐
                        │  LLM Client     │
                        │  (OpenRouter)   │
                        └─────────────────┘
```

### 5.2 Processing Steps

1. **Load Dataset**: Fetch `olimpia20/toxicity-dataset-ro-master` from HuggingFace
2. **Generate/Load Rubrics**: Use predefined rubrics (E1-E4, I1-I4, P1-P3)
3. **Evaluate Each Sample**: LLM judge evaluates text against all rubrics
4. **Calculate Rewards**: Compute explicit (weighted sum) and implicit (holistic) rewards
5. **Save Output**: Store augmented dataset in Parquet and JSON formats

### 5.3 Execution

```bash
# Install dependencies
uv sync

# Run pipeline (5 samples for testing)
uv run python main.py run --limit 5 --predefined-rubrics

# Run pipeline (full dataset)
uv run python main.py run --limit 400 --predefined-rubrics
```

---

## 6. Results

### 6.1 Pipeline Execution Statistics

```
Pipeline Statistics (5 samples):
  Total samples:        5
  Processed:            5
  Failed:               0
  Avg rubrics/sample:   11.00
  Avg explicit reward:  0.3503
  Avg implicit reward:  0.6889
  Execution time:       12.49s
```

### 6.2 Sample Output

**Input Text:** "ia uite alta mizerie politică" (translation: "look, another political mess")

**Analysis:**
- Ground Truth Label: Non-toxic (political criticism)
- Personal Attacks (E2): Not detected
- Threats (E3): Not detected
- Group Hatred (E4): Not detected
- False Positive Risk (P1): High (legitimate criticism)

**Rewards:**
- Explicit Reward: 0.2174 (low - few toxic indicators found)
- Implicit Reward: 0.8889 (high - LLM recognizes as legitimate criticism)

### 6.3 Output Files

| File | Format | Purpose |
|------|--------|---------|
| `output/augmented_dataset.parquet` | Binary | Efficient storage for ML training |
| `output/augmented_dataset.json` | JSON | Human-readable inspection |
| `rubrics.json` | JSON | Rubric documentation with examples |

### 6.4 Output Schema

```json
{
  "sample_id": 0,
  "text": "original Romanian text",
  "label": null,
  "reward_explicit": 0.3292,
  "reward_implicit": 0.3333,
  "rubrics": [
    {
      "rubric_id": "E1",
      "title": "Correct Label Assignment",
      "weight": 1.0,
      "category": "Essential",
      "evaluation_method": "direct_comparison",
      "patterns": null,
      "trigger_condition": null
    },
    // ... 10 more rubrics
  ]
}
```

---

## 7. Future Work

### 7.1 Phase 2: Full Dataset Augmentation

Process all 400 training samples to create the complete augmented dataset:

```bash
uv run python main.py run --limit 400 --predefined-rubrics
```

Estimated cost: $1-2 via OpenRouter API.

### 7.2 Phase 3: GRPO Training

Implement the reinforcement learning training loop:

```
┌─────────────────────────────────────────────────────────────────┐
│ GRPO Training Loop                                              │
├─────────────────────────────────────────────────────────────────┤
│ 1. Load augmented dataset                                       │
│ 2. Initialize base model (e.g., Llama-3.1-8B)                   │
│ 3. For each training step:                                      │
│    a. Sample batch of texts                                     │
│    b. Model generates predictions                               │
│    c. Evaluate predictions against rubrics                      │
│    d. Compute reward signal                                     │
│    e. Update model using GRPO                                   │
│ 4. Evaluate on test set                                         │
└─────────────────────────────────────────────────────────────────┘
```

### 7.3 Phase 4: Evaluation

Metrics to evaluate the trained model:

- Accuracy on toxicity classification
- False positive rate on legitimate criticism
- False negative rate on implicit toxicity
- Comparison with baseline models

---

## 8. References

1. **Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains** - Scale AI (2024)
   - Original RaR methodology paper

2. **TORCH-RaR Guide** - Project documentation
   - Section 5.1: Rubric specifications for Romanian toxicity detection

3. **Dataset**: `olimpia20/toxicity-dataset-ro-master`
   - HuggingFace: https://huggingface.co/datasets/olimpia20/toxicity-dataset-ro-master
   - 400 training samples, 100 test samples
   - Romanian political discourse

4. **LiteLLM Documentation**
   - https://docs.litellm.ai/

5. **OpenRouter API**
   - https://openrouter.ai/

---

## Appendix A: Key Code Changes

### A.1 Rubric Definitions Update

**File:** `src/torch_rar/rubric_generator.py`

**Changes:**
1. Added `evaluation_method`, `patterns`, and `trigger_condition` fields to `RubricItem` dataclass
2. Rewrote all 11 rubrics (E1-E4, I1-I4, P1-P3) to match guide Section 5.1 specifications
3. Updated weights from uniform (1.0, 0.7, -0.9) to specific values per rubric

### A.2 Configuration Update

**File:** `settings.yaml`

**Changes:**
1. Set `llm_provider: openrouter`
2. Configured API key for OpenRouter
3. Documented individual rubric weights in comments

---

## Appendix B: Commands Reference

```bash
# Setup
uv sync                                          # Install dependencies

# Test configuration
uv run python main.py test                       # Test LLM connection

# Run pipeline
uv run python main.py run --limit 5 --predefined-rubrics    # Test run
uv run python main.py run --limit 400 --predefined-rubrics  # Full run

# View output
cat output/augmented_dataset.json | python -m json.tool     # Pretty print
```

---

**Document Version:** 1.0
**Last Updated:** December 16, 2024
**Authors:** TORCH-RaR Development Team
