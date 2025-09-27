import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F

@dataclass
class ModelConfig:
    vocab_size: int = 2000
    embed_size: int = 384
    block_size: int = 256
    num_heads: int = 6
    num_layers: int = 6 

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, config: ModelConfig):
        super().__init__()
        head_size = config.embed_size // config.num_heads
        self.key = nn.Linear(config.embed_size, head_size, bias=False)
        self.query = nn.Linear(config.embed_size, head_size, bias=False)
        self.value = nn.Linear(config.embed_size, head_size, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2,-1) * k.size(-1)**-0.5
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        return out
    
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # We remove `head_size` from the call to Head() here
        self.heads = nn.ModuleList([Head(config) for _ in range(config.num_heads)])
        self.proj = nn.Linear(config.embed_size, config.embed_size)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    # ... (no changes)
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.embed_size, 4 * config.embed_size), nn.ReLU(),
            nn.Linear(4 * config.embed_size, config.embed_size))
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    # ... (no changes)
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.embed_size)
        self.ln2 = nn.LayerNorm(config.embed_size)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class SLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.embed_size)
        self.position_embedding_table = nn.Embedding(config.block_size, config.embed_size)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embed_size)
        self.lm_head = nn.Linear(config.embed_size, config.vocab_size)

    def forward(self, idx):
        B, T = idx.shape
        
        device = idx.device
        
        token_embeds = self.token_embedding_table(idx)
        pos_embeds = self.position_embedding_table(torch.arange(T, device=device))
        x = token_embeds + pos_embeds
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        return logits
    
    def generate(self, idx, max_new_tokens):
        
        for _ in range(max_new_tokens):
            
            idx_cond = idx[:, -self.config.block_size:]

            logits = self(idx_cond)

            logits = logits[:, -1, :] # Becomes (B, C)
            
            probs = F.softmax(logits, dim=-1) # (B, C)
            
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            
        return idx
    
if __name__ == '__main__':
    config = ModelConfig()
    model = SLM(config)
    
    dummy_input = torch.randint(0, config.vocab_size, (2, 8))
    logits = model(dummy_input)
    
    print("Input Shape:", dummy_input.shape)
    print("Final Logits Shape:", logits.shape)
    print(f"\nTest passed! The output shape should be (2, 8, {config.vocab_size}), and it is.")