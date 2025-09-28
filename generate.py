import torch
from model import SLM, ModelConfig
from tokenizers import Tokenizer

MODEL_PATH = "slm_model.pth" # Path to your saved model
config = ModelConfig()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


print("Loading model and tokenizer...")
model = SLM(config)
model.load_state_dict(torch.load(MODEL_PATH))
model.to(device)
model.eval() 

tokenizer = Tokenizer.from_file("tokenizer.json")

print("\nModel loaded. You can now enter prompts.")
while True:
    prompt = input("> ")
    if prompt.lower() in ['exit', 'quit']:
        break

    start_ids = tokenizer.encode(prompt).ids
    start_tensor = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)

    print("...generating...")
    with torch.no_grad():
        generated_ids = model.generate(start_tensor, max_new_tokens=100)[0].tolist()

    generated_text = tokenizer.decode(generated_ids)
    print("\n--- OUTPUT ---")
    print(generated_text)
    print("----------------\n")
