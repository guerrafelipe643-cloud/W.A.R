# W.A.R. — Modular Large Language Model Architecture

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Precision](https://img.shields.io/badge/Precision-BF16-blue)](#)
[![Architecture](https://img.shields.io/badge/Architecture-Promoter%20%E2%86%92%20Receptor%20%E2%86%92%20Judge-success)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> *Read this document in [English](#-english-documentation) or [Português](#-documentação-em-português).*

---

## 🌐 English Documentation

### Overview

**W.A.R.** is an open-source Large Language Model (LLM) and modular safety architecture built entirely from scratch in PyTorch (~1.09B combined parameters). Designed and implemented by a 14-year-old independent student researcher from Brazil, the project explores both foundational Transformer pre-training efficiency and a multi-stage **Promoter $\rightarrow$ Receptor $\rightarrow$ Judge** pipeline aimed at mitigating adversarial attacks and factual hallucinations at the architectural level.

### Architectural Highlights

#### 1. Core Transformer Backbone (`ReceptorGPT` — ~457M Parameters)
Rather than relying on high-level wrappers, W.A.R. implements modern LLM components from the ground up:

* **RMSNorm Pre-Normalization:** Root Mean Square Layer Normalization applied before attention and feed-forward sub-layers for training stability:

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma$$

* **SwiGLU Feed-Forward Network:** Gated Linear Units with Swish ($\text{SiLU}$) activation and hardware-aligned hidden dimensions ($d_{\text{ffn}} = 2816$, multiple of $128$ for TPU/GPU Tensor Core efficiency):

$$\text{SwiGLU}(x) = \left(xW_1 \odot \text{SiLU}(xW_3)\right)W_2$$

* **Flash / Scaled Dot-Product Attention:** Multi-Head Attention ($16$ heads, $d_{\text{model}} = 1024$, $24$ layers) with scaled residual initialization ($\sigma = 0.02 / \sqrt{2N_{\text{layers}}}$) and configurable causal masking (`is_causal`).
* **Weight Tying:** Shared input embedding and output projection matrices (`token_emb.weight == lm_head.weight`) using the **Qwen 2.5** tokenizer ($V = 151,643$).
* **Zero-RAM Disk Streaming (`numpy.memmap`):** Tokenizes and packs up to $2\times 10^9$ tokens from `HuggingFaceFW/fineweb` into contiguous $2049$-token sequences streamed directly from disk with virtually zero system RAM overhead.

#### 2. Tri-Modular Safety & Verification Pipeline

```text
[User Prompt] 
      │
      ▼
┌──────────────────────────────┐
│  1. PROMOTER (~229M Params)  │ ──► Bidirectional Classifier: Detects adversarial prompts / jailbreaks.
└──────────────┬───────────────┘     (Trained on Prompt-Injection datasets; blocks unsafe inputs)
               │ Approved (P >= 0.5)
               ▼
┌──────────────────────────────┐
│  2. RECEPTOR (~457M Params)  │ ──► Autoregressive Decoder GPT: Generates candidate response.
└──────────────┬───────────────┘     (Pre-trained on HuggingFaceFW/FineWeb)
               │ Candidate Output
               ▼
┌──────────────────────────────┐
│   3. JUDGE (~407M Params)    │ ──► Bidirectional Classifier: Evaluates (Prompt + Output) for hallucination.
└──────────────┬───────────────┘     (Initialized from Receptor weights + fine-tuned on HaluEval)
               │ Verified (P >= 0.5)
               ▼
       [Final Response]
```

### Model Specifications

| Component | Architectural Role | $d_{\text{model}}$ | $n_{\text{heads}}$ | $n_{\text{blocks}}$ | Context Length | Attention Mask | Parameters |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Promoter** (`WARClassifier`) | Adversarial / Safety Filter | `768` | `12` | `16` | `2048` | Bidirectional (`is_causal=False`) | **~229M** |
| **Receptor** (`GPT`) | Autoregressive Generator | `1024` | `16` | `24` | `2048` | Causal (`is_causal=True`) | **~457M** |
| **Judge** (`WARClassifier`) | Hallucination Verifier | `1024` | `16` | `20` | `2048` | Bidirectional (`is_causal=False`) | **~407M** |
| **Total System** | **Tri-Modular Pipeline** | — | — | **60** | `2048` | **Hybrid** | **~1.09B** |

### Quick Start

```bash
# Install dependencies
pip install torch numpy transformers datasets matplotlib
```

Run the training and streaming pipeline:

```bash
python train_war.py
```

---

## 🇧🇷 Documentação em Português

### Sobre o Projeto

O **W.A.R.** é um Modelo de Linguagem de Grande Escala (LLM) *open-source* construído inteiramente do zero em PyTorch (~1,09 bilhão de parâmetros somados). O projeto combina um núcleo generativo moderno (baseado nas arquiteturas LLaMA/Qwen) com uma arquitetura tripla (**Promotor $\rightarrow$ Receptor $\rightarrow$ Juiz**) desenhada para filtrar prompts adversariais e reduzir alucinações antes de entregar a resposta final ao usuário.

### Diferenciais Técnicos

* **Implementação Pura em PyTorch:** Construção manual das camadas de atenção multi-cabeça (`MultiHeadAttention`), normalização por raiz do valor quadrático médio (`RMSNorm`), rede densa ativada por porta `SwiGLU`, *Mean Pooling* mascarado para classificação e *Weight Tying* entre a matriz de Embedding e a `lm_head`.
* **Pipeline de Dados em Streaming com `numpy.memmap`:** Processamento e empacotamento (*sequence packing*) do dataset `FineWeb` em janelas de contexto de $2048$ tokens gravando diretamente em arquivos binários mapeados em disco (`.bin`), evitando estouro de memória RAM durante o pré-treinamento.
* **Transfer Learning Interno (Receptor $\rightarrow$ Juiz):** O módulo Juiz herda diretamente as representações linguísticas pré-treinadas dos primeiros 20 blocos do Receptor (`strict=False`), acelerando a convergência na detecção de alucinações.
* **Otimizado para Aceleradores (GPU CUDA & Cloud TPU):** Suporte nativo a precisão mista `bfloat16` (`torch.autocast`), compilação de grafo via `torch.compile`, *gradient clipping* e dimensões internas alinhadas em múltiplos de $128$.

### Arquitetura Tripla

1. **Promotor (~229M parâmetros):** Sub-IA classificadora binária que analisa os tokens de entrada com atenção bidirecional e decide se o prompt é seguro ou se contém tentativas de ataque adversarial (*Prompt Injection / Jailbreak*).
2. **Receptor (~457M parâmetros):** O modelo GPT autoregressivo principal que recebe os tokens aprovados pelo Promotor e gera a resposta candidata.
3. **Juiz (~407M parâmetros):** Sub-IA classificadora que avalia o par `(Prompt + Resposta)` para julgar a consistência factual e barrar alucinações.

---

## 👤 Author / Autor

**Felipe Guerra Rodrigues Athaydes**  
*Independent AI & Robotics Student Researcher (14 years old) — Anápolis, GO, Brazil*

* 🥇 **Gold Medalist (10/10 Max Score)** — Brazilian Astronomy and Astronautics Olympiad (OBA 2026)
* 🏆 **1st Place Champion** — ORCOP Robotics Olympiad
* 📫 **Contact:** [guerrafelipe643@gmail.com](mailto:guerrafelipe643@gmail.com)
* 💻 **GitHub:** [@guerrafelipe643-cloud](https://github.com/guerrafelipe643-cloud)
