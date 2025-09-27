from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

# BPE is the model algorithm.
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(vocab_size=2000, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

files = ["data/corpus.txt"]
tokenizer.train(files, trainer)

tokenizer.save("tokenizer.json")

print("Tokenizer training complete! Saved to tokenizer.json")