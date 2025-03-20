import torch
import torch.nn as nn

torch.set_default_device("mps")

param = nn.Parameter(torch.randn(10))
print(f"Parameter device: {param.device}")  # prints "Parameter device: mps:0"

optimizer = torch.optim.SGD([param], lr=1e-3)  # blows up!