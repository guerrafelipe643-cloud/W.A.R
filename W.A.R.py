import os
import math
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from datasets import load_dataset
from google.colab import drive, runtime
import numpy as np

# ==========================================================
# 1. MONTAR GOOGLE DRIVE
# ==========================================================
drive.mount("/content/drive")

SAVE_DIR = "/content/drive/MyDrive/WAR_DATA"
os.makedirs(SAVE_DIR, exist_ok=True)

train_path = f"{SAVE_DIR}/fineweb_train.pt"
val_path = f"{SAVE_DIR}/fineweb_val.pt"

# ==========================================================
# 2. TOKENIZADOR QWEN 2.5 (Definido logo no início!)
# ==========================================================
tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-1.5B",
    use_fast=True
)

vocab_size = len(tokenizer)
eos_token_id = tokenizer.eos_token_id

def encode(text):
    return tokenizer.encode(text, add_special_tokens=False)

def decode(ids):
    return tokenizer.decode(ids)

print("Tokenizador:", tokenizer.name_or_path)
print("Vocabulário:", vocab_size)
print("EOS:", eos_token_id)

# Teste rápido
teste = "Once upon a time"
ids_teste = encode(teste)
print("Teste:", teste, "-> Tokens:", ids_teste, "-> Decodificado:", decode(ids_teste))

# ==========================================================
# 3. BAIXAR E EMPACOTAR DATASET DIRETO NO DISCO (MEMMAP)
# ==========================================================
CONTEXT_LENGTH = 2048
PACK_LENGTH = CONTEXT_LENGTH + 1  # 2049 tokens (entrada + alvo)
MAX_TOKENS = 500_000_000
TOTAL_SEQS = MAX_TOKENS // PACK_LENGTH  # ~976.085 sequências de 2049

train_bin_path = f"{SAVE_DIR}/fineweb_train.bin"
val_bin_path = f"{SAVE_DIR}/fineweb_val.bin"

if not (os.path.exists(train_bin_path) and os.path.exists(val_bin_path)):
    dataset = load_dataset("HuggingFaceFW/fineweb", name="CC-MAIN-2024-10", split="train", streaming=True)

    # 99% para treino, 1% para validação
    n_val = max(1, int(TOTAL_SEQS * 0.01))
    n_train = TOTAL_SEQS - n_val

    # Cria os arquivos direto no disco (int32 suporta os 151.643 tokens do Qwen)
    train_mmap = np.memmap(train_bin_path, dtype=np.int32, mode="w+", shape=(n_train, PACK_LENGTH))
    val_mmap = np.memmap(val_bin_path, dtype=np.int32, mode="w+", shape=(n_val, PACK_LENGTH))

    buffer = []
    docs_processados = 0
    seqs_salvas = 0

    for example in dataset:
        text = example["text"]
        if not text or len(text.strip()) < 100:
            continue

        ids = encode(text) + [eos_token_id]
        buffer.extend(ids)
        docs_processados += 1

        # Sempre que o buffer atinge 2049 tokens, grava direto no disco!
        while len(buffer) >= PACK_LENGTH and seqs_salvas < TOTAL_SEQS:
            seq = buffer[:PACK_LENGTH]
            buffer = buffer[PACK_LENGTH:]

            if seqs_salvas < n_train:
                train_mmap[seqs_salvas] = seq
            else:
                val_mmap[seqs_salvas - n_train] = seq

            seqs_salvas += 1

        if docs_processados % 2000 == 0:
            # Descarrega o cache para o disco periodicamente
            train_mmap.flush()
            val_mmap.flush()
            print(
                f"Docs: {docs_processados:,} | "
                f"Sequências gravadas no disco: {seqs_salvas:,}/{TOTAL_SEQS:,} "
                f"({(seqs_salvas / TOTAL_SEQS) * 100:.2f}%)"
            )

        if seqs_salvas >= TOTAL_SEQS:
            print("Limite de sequências atingido!")
            break

    train_mmap.flush()
    val_mmap.flush()
    print(f"Dataset salvo no disco com sucesso! Treino: {n_train:,} | Val: {n_val:,}")
else:
    print("\nArquivos .bin já encontrados no Drive! Pulando download...")
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

class WARTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32)
        }

# ==========================================================
# 1. PREPARANDO DADOS DO PROMOTOR (1.0 = Seguro, 0.0 = Ataque)
# ==========================================================
ds_prom = load_dataset("deepset/prompt-injections", split="train")
textos_promotor = [ex["text"] for ex in ds_prom]
# No dataset original: 0 é seguro e 1 é ataque. Invertemos para 1.0 = Seguro:
labels_promotor = [1.0 if ex["label"] == 0 else 0.0 for ex in ds_prom]

promotor_loader = DataLoader(
    WARTextDataset(textos_promotor, labels_promotor, tokenizer, max_len=512),
    batch_size=8,
    shuffle=True
)

# ==========================================================
# 2. PREPARANDO DADOS DO JUIZ (1.0 = Factual, 0.0 = Alucinação)
# ==========================================================
ds_juiz = load_dataset("pminervini/HaluEval", "qa", split="data")
textos_juiz = []
labels_juiz = []

# Pegamos 10.000 exemplos (5.000 verdadeiros + 5.000 alucinações)
for ex in ds_juiz.select(range(5000)):
    pergunta = ex["question"]
    # Exemplo Verdadeiro (1.0)
    textos_juiz.append(f"Question: {pergunta}\nAnswer: {ex['right_answer']}")
    labels_juiz.append(1.0)
    # Exemplo com Alucinação (0.0)
    textos_juiz.append(f"Question: {pergunta}\nAnswer: {ex['hallucinated_answer']}")
    labels_juiz.append(0.0)

juiz_loader = DataLoader(
    WARTextDataset(textos_juiz, labels_juiz, tokenizer, max_len=512),
    batch_size=8,
    shuffle=True
)

print(f"Batches do Promotor: {len(promotor_loader)} | Batches do Juiz: {len(juiz_loader)}")
# ==========================================================
# 4. CONFIGURAR DISPOSITIVO (TPU / GPU / CPU)
# ==========================================================
try:
    import torch_tpu
    device = torch.device("tpu")
    use_bf16 = True
    print("Dispositivo TPU:", torch.tpu.get_device_properties().name)
except Exception:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = torch.cuda.is_bf16_supported() if device.type == "cuda" else False
    print(f"Dispositivo: {device} | BF16 habilitado: {use_bf16}")

# ==========================================================
# 5. ARQUITETURA DO MODELO WAR (GPT)
# ==========================================================
d_model = 1024
context_length = 2048
n_heads = 16
n_blocks = 24
class SwiGLUFFN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # Geralmente usa-se 8/3 ou 4 vezes o tamanho do d_model
        hidden_dim = 2816
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

    def forward(self, x):
        # Fórmula do SwiGLU: (xW1 * SiLU(xW3)) * W2
        return self.w2(self.w1(x) * torch.nn.functional.silu(self.w3(x)))
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        # O peso (gamma) que o modelo aprende durante o treino para escalar a saída
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # 1. Calcula o quadrado de x
        # 2. Tira a média na última dimensão (d_model)
        # 3. Soma um epsilon minúsculo para nunca dividir por zero
        # 4. Tira a raiz quadrada invertida (rsqrt é mais rápido na GPU)
        norm_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        return norm_x * self.weight
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, debug=False):
        super().__init__()
        self.debug = debug
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, is_causal = True):
        B, T, C = x.shape
        Q = self.q_proj(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).reshape(B, T, self.n_heads, self.d_head).transpose(1, 2)

        out = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, is_causal=is_causal
        )
        out = out.transpose(1, 2).contiguous().reshape(B, T, C)
        out = self.out_proj(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model)
    def forward(self, x, is_causal = True):
        x = x + self.mha(self.ln1(x), is_causal = is_causal)
        x = x + self.ffn(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, d_model, n_heads, n_blocks, vocab_size, context_length):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_length, d_model)
        self.ln_f = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_blocks)]
        )
        self.init_weights()
        # weight tying: lm_head e token_emb compartilham pesos
        self.lm_head.weight = self.token_emb.weight

    def forward(self, tokens):
        B, T = tokens.shape
        positions = torch.arange(T, device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

        # Escala das projeções residuais (Corrigido o símbolo de *)
        residual_std = 0.02 / math.sqrt(2 * n_blocks)
        for block in self.blocks:
            torch.nn.init.normal_(block.mha.out_proj.weight, mean=0.0, std=residual_std)
            torch.nn.init.normal_(block.ffn.w2.weight, mean=0.0, std=residual_std)
class WARClassifier(nn.Module):
    def __init__(self, d_model, n_heads, n_blocks, vocab_size, context_length, pad_token_id):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context_length, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads) for _ in range(n_blocks)]
        )      
        self.ln_f = RMSNorm(d_model)
        self.classifier_head = nn.Linear(d_model, 1, bias=False)   
        self.init_weights(n_blocks)
    def init_weights(self, n_blocks):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        residual_std = 0.02 / math.sqrt(2 * n_blocks)
        for block in self.blocks:
            torch.nn.init.normal_(block.mha.out_proj.weight, mean=0.0, std=residual_std)
            torch.nn.init.normal_(block.ffn.w2.weight, mean=0.0, std=residual_std)
    def forward(self, tokens):
        B, T = tokens.shape
        positions = torch.arange(T, device=tokens.device)
        x = self.token_emb(tokens) + self.pos_emb(positions)
        for block in self.blocks:
            x = block(x, is_causal=False)
        x = self.ln_f(x)
        mask = (tokens != self.pad_token_id).unsqueeze(-1).type_as(x)
        summed = (x * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-6)
        pooled = summed / counts
        logit = self.classifier_head(pooled).squeeze(-1) 
        return logit
gpt = GPT(d_model, n_heads, n_blocks, vocab_size, context_length).to(device)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
pad_id = tokenizer.pad_token_id
print("Parâmetros do Receptor:", f"{sum(p.numel() for p in gpt.parameters()):,}")
tokens_teste = torch.randint(0, vocab_size, (2, 64), device=device)
logits_teste = gpt(tokens_teste)
assert logits_teste.shape == (2, 64, vocab_size)
print("WAR: teste estrutural OK | Parâmetros:", f"{sum(p.numel() for p in gpt.parameters()):,}")
train_tokens = np.memmap(train_bin_path, dtype=np.int32, mode="r").reshape(-1, PACK_LENGTH)
val_tokens = np.memmap(val_bin_path, dtype=np.int32, mode="r").reshape(-1, PACK_LENGTH)

print("Train shape:", train_tokens.shape)
print("Val shape:", val_tokens.shape)

batch_size = 8   # Se tiver bastante memória na TPU/GPU, pode usar 16
epochs = 1
if device.type in ("cuda", "tpu", "xla"):
    use_bf16 = True
gpt = torch.compile(gpt)

print("Modelo compilado!")
optimizer = torch.optim.AdamW(
    gpt.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    fused=(device.type == "cuda")
)

@torch.no_grad()
def evaluate(model, val_data, batch_size, vocab_size, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    for i in range(0, len(val_data) - batch_size, batch_size):
        batch = torch.from_numpy(np.array(val_data[i : i + batch_size], dtype=np.int64)).to(device)
        inputs = batch[:, :-1]  # (batch_size, 2048)
        targets = batch[:, 1:]  # (batch_size, 2048)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1)
            )

        total_loss += loss.item()
        total_batches += 1

    model.train()
    return total_loss / max(1, total_batches)

losses = []
val_losses = []
step = 0  # <--- Inicializado antes do loop!

gpt.train()
# Exemplo de função de treino para o Promotor ou Juiz
def treinar_classificador(modelo, dataloader, epochs=1, lr=1e-4):
    optimizer = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=0.01)
    # Essa função de Loss já aplica a Sigmoid internamente com precisão matemática máxima
    criterion = nn.BCEWithLogitsLoss()
    
    modelo.train()
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            # 1. Pegamos os tokens da frase (B, T) e o rótulo verdadeiro 0.0 ou 1.0 (B,)
            input_ids = batch["input_ids"].to(device, dtype=torch.long)
            labels = batch["labels"].to(device, dtype=torch.float32)

            # 2. Zeramos os gradientes acumulados do passo anterior
            optimizer.zero_grad(set_to_none=True)

            # 3. Forward Pass em BF16
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                # O modelo devolve 1 nota (logit) para cada frase do batch
                logits = modelo(input_ids)
                # Comparamos a nota do modelo com o gabarito (0.0 ou 1.0)
                loss = criterion(logits, labels)

            # 4. Backward Pass: calcula o erro de cada peso
            loss.backward()
            
            # 5. Evita explosão de gradiente
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            
            # 6. Atualiza os pesos
            optimizer.step()

            if step % 50 == 0:
                # Calculamos a acurácia na hora: se logit > 0.0, o modelo chutou 1.0!
                preds = (logits > 0.0).float()
                acc = (preds == labels).float().mean().item() * 100
                print(f"Step {step} | Loss: {loss.item():.4f} | Acurácia: {acc:.1f}%")
for epoch in range(epochs):
    print(f"\n===== EPOCH {epoch + 1}/{epochs} =====")

    # Loop único em batches 2D (sem o loop 'start' duplicado)
    for i in range(0, len(train_tokens) - batch_size, batch_size):
        batch = torch.from_numpy(np.array(train_tokens[i : i + batch_size], dtype=np.int64)).to(device)
        inputs = batch[:, :-1]  # (batch_size, 2048)
        targets = batch[:, 1:]  # (batch_size, 2048)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            logits = gpt(inputs)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1)
            )

        # 4. Calcula os gradientes (Essencial! Estava faltando no seu código)
        loss.backward()

        # 5. Corta gradientes explosivos (evita Loss = NaN em modelos grandes)
        torch.nn.utils.clip_grad_norm_(gpt.parameters(), max_norm=1.0)

        # 6. Atualiza os pesos
        optimizer.step()
        # Sincroniza a cada 50 steps para não deixar a TPU lenta
        if step % 50 == 0:
            loss_val = loss.item()
            losses.append(loss_val)
            progresso = (i / len(train_tokens)) * 100
            print(
                f"Step {step:,} | "
                f"Sequências: {i:,}/{len(train_tokens):,} ({progresso:.2f}%) | "
                f"Loss: {loss_val:.4f}"
            )
        # Salva um backup automático no Drive a cada 1.000 steps!
        if step > 0 and step % 1000 == 0:
            modelo_backup = gpt._orig_mod if hasattr(gpt, "_orig_mod") else gpt
            torch.save(modelo_backup.state_dict(), f"{SAVE_DIR}/WAR_receptor_backup.pt")
            print(f"---> Backup salvo no Drive no step {step}!")
        step += 1

    val_loss = evaluate(gpt, val_tokens, batch_size, vocab_size, device)
    val_losses.append(val_loss)
    print(f"Validation Loss: {val_loss:.4f}")

print("\nTreinamento terminado!")
promotor = WARClassifier(
    d_model=768,
    n_heads=12,
    n_blocks=16,
    vocab_size=vocab_size,
    context_length=2048,
    pad_token_id=pad_id
).to(device)
juiz = WARClassifier(
    d_model=1024,
    n_heads=16,
    n_blocks=20,
    vocab_size=vocab_size,
    context_length=2048,
    pad_token_id=pad_id
).to(device)
# Copia os Embeddings e os primeiros 20 blocos do Receptor pré-treinado para o Juiz!
receptor_state = gpt._orig_mod.state_dict() if hasattr(gpt, "_orig_mod") else gpt.state_dict()
juiz_state = juiz.state_dict()

for k, v in receptor_state.items():
    if k in juiz_state and juiz_state[k].shape == v.shape:
        juiz_state[k].copy_(v)
treinar_classificador(promotor, promotor_loader, epochs=1, lr=1e-4)
treinar_classificador(juiz, juiz_loader, epochs=1, lr=1e-4)
# ==========================================================
# 7. GRÁFICO E SALVAMENTO DO MODELO
# ==========================================================
plt.plot(losses, label="Training Loss (a cada 50 steps)")
if val_losses:
    plt.axhline(y=val_losses[-1], color="r", linestyle="--", label="Validation Loss")
plt.xlabel("Logs (x50 Steps)")
plt.ylabel("Loss")
plt.title("WAR Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()

Path("/content/drive/MyDrive/").mkdir(exist_ok=True)
config_dict = {
    "vocab_size": vocab_size,
    "d_model": d_model,
    "context_length": context_length,
    "n_heads": n_heads,
    "n_blocks": n_blocks,
}
receptor_to_save = gpt._orig_mod if hasattr(gpt, "_orig_mod") else gpt
promotor_to_save = promotor._orig_mod if hasattr(promotor, "_orig_mod") else promotor
juiz_to_save = juiz._orig_mod if hasattr(juiz, "_orig_mod") else juiz

torch.save(receptor_to_save.state_dict(), "/content/drive/MyDrive/WAR2_receptor.pt")
torch.save(promotor_to_save.state_dict(), "/content/drive/MyDrive/WAR2_promotor.pt")
torch.save(juiz_to_save.state_dict(), "/content/drive/MyDrive/WAR2_juiz.pt")
torch.save(
    {
        "model_state_dict": receptor_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config_dict,
        "epoch": epoch,
    },
    "/content/drive/MyDrive/WAR2_checkpoint.pt",
)

print("WAR salvo com sucesso!")
print("Desligando a GPU automaticamente para economizar suas 80 unidades!")
runtime.unassign()
