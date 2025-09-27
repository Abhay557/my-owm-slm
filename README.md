# SLM-From-Scratch: Building a Small Language Model for AI Theory  

This repository contains a **Small Language Model (SLM)** built entirely from scratch using **Python** and **PyTorch**.  
This isn’t about creating the next ChatGPT — it’s a **deep dive project** to demystify the **core components** of modern language models.  
Inside, you’ll find a **decoder-only Transformer**, trained on a custom corpus of AI theory papers, built step by step to understand the technology powering today’s AI revolution.  

---

##  Features  
- **Custom Data Pipeline**: Extract & preprocess text from PDFs.  
- **Custom Tokenizer**: Byte-Pair Encoding (BPE) tokenizer trained on AI corpus.  
- **Transformer Architecture**: From scratch implementation with:  
  - Token + Positional Embeddings  
  - Multi-Head Self-Attention  
  - Feed-Forward Networks  
  - Residual Connections & LayerNorm  
- **Training & Evaluation Loop**:  
  - AdamW optimizer  
  - Cross-entropy loss  
  - Validation loop for overfitting checks  
  - GPU acceleration (CUDA)  
- **Autoregressive Text Generation**: Generate new text from a prompt.  

---

##  Skills Demonstrated  
- **PyTorch Proficiency**: Custom `nn.Module`, tensors, training loops.  
- **NLP Fundamentals**: Tokenization, embeddings, language modeling.  
- **Transformer Architecture**: Hands-on implementation of self-attention & residuals.  
- **Data Engineering**: Cleaning, corpus prep, tokenization pipeline.  
- **Software Best Practices**: Organized modules, venvs, dependencies, docs.  
- **GPU Computing**: CUDA acceleration & device management.  

---

##  Key Design Decisions  
- **Tokenizer**: BPE → robust subword representation.  
- **Architecture**: Decoder-only Transformer → GPT-like autoregressive text gen.  
- **Optimizer**: AdamW → better generalization than Adam/SGD.  

---

---

## How It Works (4 Phases)  

**Phase 1: Data Prep**  
- Gather AI theory texts → extract → clean → `corpus.txt`  
- Train custom `tokenizer.json`  

**Phase 2: Model Architecture**  
- Build decoder-only Transformer in `model.py`  
- Add embeddings, attention, FFN, residuals  

**Phase 3: Training**  
- Implement forward → loss → backprop → optimizer step  
- Add validation loop & CUDA support  

**Phase 4: Inference**  
- Implement autoregressive `generate()`  
- Predict token by token from a prompt  

---

## Setup & Usage  

### 1. Prerequisites  
- Python 3.8+  
- NVIDIA GPU + CUDA 11.8+  
- Git  

### 2. Installation  
```bash
# Clone repo
git clone https://github.com/your-username/SLM-From-Scratch.git
cd SLM-From-Scratch

# Virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Dependencies
pip install -r requirements.txt
```
## ✨ Example Output  

Prompt: *“The attention mechanism”*  

```text
The att ent ion mechan ism ent image ent ent mechan ent mechan 
ent mechan ent mechan ent ent ci als ent mechan ics ent ent 
negative gression ent mechan ent mechan ent giv im ilar ent ency 
ent world als between ent cript Hidden mechan ism ent ent ent 
ent Des mechan ent agent ent ent ent mechan ism ent ent ent 
ential ent ST ent day ent world who ren High struct ent ent 
ent day ent facts ent text expres mechan ent mechan ent 
ometimes Knowledge ial ent mechan ics ent ent ency ent vide 
Gener ific ent mechan ics
```
This output demonstrates that the model has captured token patterns, repetitions, and domain-specific vocabulary from the AI theory corpus.
Although the coherence is limited due to small scale and short training, the results show the model is successfully learning the statistical structure of language.
