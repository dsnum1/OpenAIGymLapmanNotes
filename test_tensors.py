import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device',device)
a = torch.rand((2,4), device=device, dtype=torch.float32)

b = torch.rand((2,4), dtype=torch.float32, device=device)


print("a", a)
print("b", b)

print('a+b',a+b)