import torch
import torch.nn.functional as F
from model import SLM, ModelConfig
from tokenizers import Tokenizer

# --- Hyperparameters ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MAX_ITERS = 500 # Let's train a bit longer for a better result
EVAL_INTERVAL = 100
LOG_INTERVAL = 10

config = ModelConfig()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = torch.load('data.pt')
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (BATCH_SIZE,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(200)
        for k in range(200):
            X, Y = get_batch(split)
            logits = model(X)
            loss = F.cross_entropy(logits.view(-1, config.vocab_size), Y.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- Main training script ---
if __name__ == '__main__':
    model = SLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print("\nStarting training on device:", device)
    for step in range(MAX_ITERS):
        if step % EVAL_INTERVAL == 0 and step > 0:
            losses = estimate_loss()
            print(f"---- Step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} ----")
        xb, yb = get_batch('train')
        logits = model(xb)
        loss = F.cross_entropy(logits.view(-1, config.vocab_size), yb.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % LOG_INTERVAL == 0:
            print(f"Step {step}: batch train loss {loss.item():.4f}")

    # --- NEW: SAVE THE MODEL'S WEIGHTS ---
    MODEL_SAVE_PATH = "slm_model.pth"
    print(f"\nTraining finished! Saving model to {MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
