# Fine-Tuning Methods for 2x H200 GPUs

Your setup is quite powerful: **2x H200 = ~282GB HBM3e VRAM** with 4.8 TB/s bandwidth per GPU. Here's what you can do:

---

## Hardware Capacity Overview

| Model Size | Full Fine-Tune | LoRA | QLoRA |
|------------|----------------|------|-------|
| 7B | ✅ Easy | ✅ | ✅ |
| 13B | ✅ Easy | ✅ | ✅ |
| 70B | ✅ Possible | ✅ | ✅ |
| 110B+ | ⚠️ With optimizations | ✅ | ✅ |
| 405B | ❌ (need offloading) | ✅ With ZeRO | ✅ |

---

## 1. Full Fine-Tuning

With ~282GB VRAM, you can **fully fine-tune models up to ~70B parameters** without offloading (rule of thumb: ~16GB per billion parameters for full fine-tuning with optimizer states).

**Best for**: Maximum performance when you have enough compute and want task-specific optimization.

---

## 2. Parameter-Efficient Methods

### LoRA (Low-Rank Adaptation)
- Adds small trainable matrices to frozen base weights
- **Memory**: ~12-24GB for 7B models
- **Performance**: 95-99% of full fine-tuning quality
- [LoRA & QLoRA Guide](https://medium.com/rebooted-minds/lora-qlora-the-beginner-friendly-guide-to-llm-fine-tuning-ce3bdb3c03e3)

### QLoRA (Quantized LoRA)
- Base model in 4-bit, LoRA adapters in higher precision
- **Memory**: ~1 byte/parameter (70B model in ~70GB)
- Enables fine-tuning **405B models** on your setup
- [QLoRA Guide](https://towardsdatascience.com/qlora-how-to-fine-tune-an-llm-on-a-single-gpu-4e44d6b5be32/)

### Spectrum (SNR-Based Layer Selection)
- Freezes low signal-to-noise layers, trains high-SNR layers only
- **Spectrum-25**: 23% less memory, 37% faster training
- **Best for**: Distributed setups with FSDP/DeepSpeed
- Combines well with QLoRA for additional savings
- [Spectrum on HuggingFace](https://huggingface.co/blog/anakin87/spectrum)

---

## 3. Distributed Training Strategies

### DeepSpeed ZeRO

| Stage | What's Sharded | Use Case |
|-------|----------------|----------|
| ZeRO-1 | Optimizer states | Memory reduction |
| ZeRO-2 | + Gradients | More memory savings |
| ZeRO-3 | + Parameters | Maximum sharding |

### FSDP (Fully Sharded Data Parallel)
- PyTorch-native alternative to DeepSpeed
- Full sharding similar to ZeRO-3
- [FSDP vs DeepSpeed](https://parlance-labs.com/education/fine_tuning/zach.html)

### CPU/NVMe Offloading
Your large RAM and SSD enable offloading:
- **Optimizer Offload**: Offload gradients/optimizer states to CPU
- **Param Offload**: Offload model parameters to CPU/NVMe
- Trade-off: Slower training, but enables larger models

---

## 4. Alignment/Preference Tuning

| Method | Reference Model | Reward Model | Stages | Best For |
|--------|-----------------|--------------|--------|----------|
| **RLHF/PPO** | Required | Required | 3 (SFT→RM→PPO) | Maximum control, code tasks |
| **DPO** | Required | ❌ | 2 (SFT→DPO) | Simpler, stable training |
| **ORPO** | ❌ | ❌ | 1 (unified) | Most efficient, memory-friendly |

- [DPO vs RLHF Comparison](https://medium.com/@baicenxiao/rlhf-vs-dpo-choosing-the-method-for-llm-alignment-tuning-66f45ef3d4b5)
- [ORPO Overview](https://argilla.io/blog/mantisnlp-rlhf-part-8/)

---

## 5. Speed Optimization Frameworks

### Unsloth
- **2-5x faster** than Flash Attention 2
- **80% less VRAM** for same workloads
- Custom Triton kernels with 0% accuracy loss
- Supports: LoRA, QLoRA, full fine-tuning, GRPO, DPO
- [Unsloth GitHub](https://github.com/unslothai/unsloth)

### Additional Optimizations
- **Flash Attention 2**: Faster attention computation
- **Gradient Checkpointing**: Trade compute for memory
- **Mixed Precision (bf16/fp16)**: Halves memory for weights
- **Liger Kernels**: Additional kernel optimizations

---

## Recommended Combinations for Your Setup

### For 70B Models (Full Power)
```
DeepSpeed ZeRO-3 + Full Fine-Tuning + Flash Attention + bf16
```

### For 100B+ Models
```
QLoRA + DeepSpeed ZeRO-3 + Gradient Checkpointing + CPU Offload
```

### For Maximum Speed (7B-13B)
```
Unsloth + LoRA + Packing + Flash Attention
→ Up to 10x faster on single GPU
```

### For Alignment Tasks
```
ORPO (single-stage) or DPO + LoRA + Unsloth
→ Memory efficient, no reward model needed
```

---

## Practical Recommendations

1. **Start with Unsloth + QLoRA** for rapid iteration
2. **Use FSDP or DeepSpeed ZeRO-3** for multi-GPU training
3. **Try Spectrum-50** to match full fine-tuning with less compute
4. **For alignment**: ORPO is simplest; DPO for more control; PPO for maximum quality
5. **Leverage your RAM/SSD** for CPU offloading on 100B+ models

---

## Sources

- [Fine-tuning LLMs in 2025 with HuggingFace](https://www.philschmid.de/fine-tune-llms-in-2025)
- [NVIDIA H200 Overview](https://www.nvidia.com/en-us/data-center/h200/)
- [GPU Options for Fine-tuning](https://www.digitalocean.com/resources/articles/gpu-options-finetuning)
- [LLM Fine-tuning GPU Guide](https://www.runpod.io/blog/llm-fine-tuning-gpu-guide)
- [Distributed Training Guide](https://sumanthrh.com/post/distributed-and-efficient-finetuning/)
- [DPO 72B Fine-Tuning](https://techcommunity.microsoft.com/blog/machinelearningblog/dpo-72b-model-fine-tuning-with-deepspeed-and-fsdp/4359713)
- [Spectrum Paper](https://arxiv.org/abs/2406.06623)
- [Unsloth Documentation](https://docs.unsloth.ai/)
