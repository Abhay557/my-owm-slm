import torch
from tokenizers import Tokenizer

tokenizer_path = "tokenizer.json"
tokenizer = Tokenizer.from_file(tokenizer_path)

corpus_path = "data/corpus.txt"
with open(corpus_path, 'r', encoding='utf-8') as f:
    text = f.read()

encoding = tokenizer.encode(text)
token_ids = encoding.ids

print(f"Corpus has been tokenized into {len(token_ids)} tokens.")

data = torch.tensor(token_ids, dtype=torch.long)


output_path = 'data.pt'
torch.save(data, output_path)

print(f"Tokenized data saved to {output_path}")