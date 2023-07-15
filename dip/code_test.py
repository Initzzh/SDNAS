import torch

# a = torch.randn(1,3,512,512)
a = torch.tensor([1.0,2.0],requires_grad=True)
print(a)
b = a.detach().clone()
print(b)
