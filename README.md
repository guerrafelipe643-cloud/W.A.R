# W.A.R
W.A.R. — Modular Large Language Model ArchitectureRead this document in English or Português.🌐 English DocumentationOverviewW.A.R. is an open-source Large Language Model (LLM) and modular safety architecture built entirely from scratch in PyTorch. Designed and implemented by a 14-year-old independent student researcher from Brazil, the project explores both foundational Transformer pre-training efficiency and a multi-stage Promoter $\rightarrow$ Receptor $\rightarrow$ Judge inference pipeline aimed at mitigating adversarial attacks and factual hallucinations at the architectural level.Architectural Highlights1. Core Transformer Backbone (ReceptorGPT — ~457M Parameters)Rather than relying on high-level wrappers, W.A.R. implements modern LLM components from the ground up:RMSNorm Pre-Normalization: Root Mean Square Layer Normalization applied before attention and feed-forward sub-layers for training stability:
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma$$SwiGLU Feed-Forward Network: Gated Linear Units with Swish ($\text{SiLU}$) activation and hardware-aligned hidden dimensions ($d_{\text{ffn}} = 2816$, multiple of $128$ for TPU/GPU Tensor Core efficiency):
$$\text{SwiGLU}(x) = \left(xW_1 \odot \text{SiLU}(xW_3)\right)W_2$$Flash / Scaled Dot-Product Causal Attention: Multi-Head Attention ($16$ heads, $d_{\text{model}} = 1024$, $24$ layers) with scaled residual initialization ($\sigma = 0.02 / \sqrt{2N_{\text{layers}}}$).Weight Tying: Shared embedding and output projection matrices (token_emb.weight == lm_head.weight) using the Qwen 2.5 tokenizer ($V = 151,643$).Zero-RAM Disk Streaming (numpy.memmap): Tokenizes and packs up to $2\times 10^9$ tokens from HuggingFaceFW/FineWeb into contiguous $2049$-token sequences streamed directly from disk with virtually zero system RAM overhead.2. Tri-Modular Safety & Verification Pipeline (Roadmap)[User Prompt] 
      │
      ▼
┌──────────────────────────────┐
│  1. PROMOTER (~229M Params)  │ ──► Binary Classifier: Detects adversarial prompts / jailbreaks.
└──────────────┬───────────────┘     (Blocks unsafe inputs before generation)
               │ Approved
               ▼
┌──────────────────────────────┐
│  2. RECEPTOR (~457M Params)  │ ──► Autoregressive Decoder GPT: Generates candidate response.
└──────────────┬───────────────┘
               │ Candidate Output
               ▼
┌──────────────────────────────┐
│   3. JUDGE (~407M Params)    │ ──► Binary Classifier: Evaluates (Prompt + Output) for hallucination.
└──────────────┬───────────────┘     (Triggers regeneration or fallback if hallucinated)
               │ Verified
               ▼
       [Final Response]
Model SpecificationsComponentRoledmodel​nheads​nblocks​Context LengthParametersPromoterAdversarial / Safety Filter$768$$12$$16$$2048$~229MReceptor (Core)Autoregressive Generator$1024$$16$$24$$2048$~457MJudgeFactual / Hallucination Verifier$1024$$16$$20$$2048$~407MQuick Start# Install dependencies
pip install torch numpy transformers datasets matplotlib
Run the training and streaming pipeline:python train_war.py
🇧🇷 Documentação em PortuguêsSobre o ProjetoO W.A.R. é um Modelo de Linguagem de Grande Escala (LLM) open-source construído inteiramente do zero em PyTorch. O projeto combina um núcleo generativo moderno (baseado nas arquiteturas LLaMA/Qwen) com uma proposta de arquitetura tripla (Promotor $\rightarrow$ Receptor $\rightarrow$ Juiz) desenhada para filtrar prompts adversariais e reduzir alucinações antes de entregar a resposta final ao usuário.Diferenciais TécnicosImplementação Pura em PyTorch: Construção manual das camadas de atenção multi-cabeça (MultiHeadAttention), normalização por raiz do valor quadrático médio (RMSNorm), rede densa ativada por porta SwiGLU e Weight Tying entre a matriz de Embedding e a lm_head.Pipeline de Dados em Streaming com numpy.memmap: Processamento e empacotamento (sequence packing) do dataset FineWeb ($2$ bilhões de tokens em janelas de contexto de $2048$ tokens) gravando diretamente em arquivos binários mapeados em disco (.bin), evitando estouro de memória RAM durante o pré-treinamento.Otimizado para Aceleradores (GPU CUDA & Cloud TPU): Suporte nativo a precisão mista bfloat16 (torch.autocast), compilação de grafo via torch.compile, gradient clipping e dimensões internas alinhadas em múltiplos de $128$.Arquitetura Tripla PlanejadaPromotor (~229M parâmetros): Sub-IA classificadora binária que analisa os tokens de entrada e decide se o prompt é seguro ou se contém tentativas de ataque adversarial (Prompt Injection / Jailbreak).Receptor (~457M parâmetros): O modelo GPT autoregressivo principal que recebe os tokens aprovados pelo Promotor e gera a resposta.Juiz (~407M parâmetros): Sub-IA classificadora que avalia o par (Prompt + Resposta) para julgar a consistência factual e barrar alucinações.👤 Author / AutorFelipe GuerraIndependent AI & Robotics Student Researcher (14 years old) — Brazil🥇 Gold Medalist (10/10 Max Score) — Brazilian Astronomy and Astronautics Olympiad (OBA 2026)🏆 1st Place Champion — ORCOP Robotics Olympiad📫 Contact: guerrafelipe643@gmail.com💻 GitHub: @guerrafelipe643-cloud
