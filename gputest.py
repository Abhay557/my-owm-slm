import torch
print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"PyTorch's CUDA version: {torch.version.cuda}")

# i was dum trainig it on cpu then i found out i can use gpu
# so i changed device to cuda in train.py