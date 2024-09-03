import torch

print("Is GPU available?", torch.cuda.is_available())  # Should return True if GPU is available
print("CUDA version PyTorch is using:", torch.version.cuda)         # Should print the CUDA version PyTorch is using

num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")

current_device = torch.cuda.current_device()
print(f"Currently selected GPU: {torch.cuda.get_device_name(current_device)}")