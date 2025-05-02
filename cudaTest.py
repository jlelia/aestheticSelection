import torch
print("CUDA available:", torch.cuda.is_available())
print("CUDA version seen by PyTorch:", torch.version.cuda)
x = torch.rand(2, device="cuda")
print("Device for x:", x.device)
