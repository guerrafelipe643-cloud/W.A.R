---
language:
  - en
  - pt
license: mit
library_name: pytorch
tags:
  - pytorch
  - llm
  - transformer
  - from-scratch
  - modular-ai
  - safety
  - hallucination-detection
  - prompt-injection
  - swiglu
  - rmsnorm
  - tpu
datasets:
  - HuggingFaceFW/fineweb
  - deepset/prompt-injections
  - pminervini/HaluEval
base_model_relation: finetune
pipeline_tag: text-generation
---

# ⚡ W.A.R. — Modular Multi-Receptor LLM & Safety Architecture

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-From%20Scratch-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Precision](https://img.shields.io/badge/Precision-BF16%20%7C%20Torch.Compile-blue.svg)]()
[![Dataset](https://img.shields.io/badge/Pre--Training-HuggingFaceFW%2Ffineweb-FFD21E.svg)](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
[![Hardware](https://img.shields.io/badge/Hardware-Cloud%20TPU%20%26%20CUDA-4285F4.svg)]()

> 🌍 **Read this documentation in:** [English](#-english-documentation) | [Português](#-documentação-em-português)

---

## 🌐 English Documentation

### Overview
**W.A.R.** is an open-source Large Language Model (LLM) and modular verification ecosystem built **entirely from scratch in pure PyTorch**. Designed and implemented by a 14-year-old independent student researcher from Brazil, the project addresses two of the most critical challenges in modern foundation models — **adversarial prompt injections** and **factual hallucinations** — directly at the architectural level rather than relying on external API wrappers.

The baseline tri-modular pipeline comprises **~1.09B parameters** (1 Promoter + 1 Receptor + 1 Judge) and scales up to **~2.92B parameters** when instantiated with **5 domain-specialized Receptors** via CPU-to-Device dynamic memory swapping.

---

### 🏗️ Architectural Pipeline (Promoter $\rightarrow$ 5x Receptors $\rightarrow$ Judge)

```text
                             [ User Input Prompt ]
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. PROMOTER (~229M Params | Bidirectional Transformer Encoder)              │
│    • Inspects full prompt context (is_causal=False) + Masked Mean Pooling   │
│    • Trained on deepset/prompt-injections (1.0 = Safe, 0.0 = Attack)        │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                        If Safe (P >= 0.5) & Routed
                                       │
         ┌──────────────────┬──────────┴────────┬──────────────────┐
         ▼                  ▼                   ▼                  ▼
  ┌─────────────┐    ┌─────────────┐     ┌─────────────┐    ┌─────────────┐
  │ Receptor #0 │    │ Receptor #1 │ ... │ Receptor #3 │    │ Receptor #4 │
  │ (~457M GPT) │    │ (~457M GPT) │     │ (~457M GPT) │    │ (~457M GPT) │
  └──────┬──────┘    └──────┬──────┘     └──────┬──────┘    └──────┬──────┘
         └──────────────────┴──────────┬────────┴──────────────────┘
                                       │
                     Candidate Response (Dynamic Device Swap)
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. JUDGE (~407M Params | Bidirectional Hallucination Verifier)              │
│    • Warm-started from pretrained Receptor Embeddings + First 20 Blocks     │
│    • Evaluates (Question + Candidate Answer) on pminervini/HaluEval         │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                         If Factual (P >= 0.5)
                                       ▼
                            [ Verified Final Output ]
