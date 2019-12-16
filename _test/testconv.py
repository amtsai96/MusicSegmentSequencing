import torch
from torch import nn

a = torch.randn(64, 1, 100, 128)  
#m = nn.Conv1d(12, 12, 3) 
m=nn.Conv2d(1, 70, kernel_size=(3, 128))
out = m(a)
print(out.size())#(64, 12, 98)
print(m)