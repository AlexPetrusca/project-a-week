from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

torch.set_default_device("mps")
torch.manual_seed(1337)

@dataclass
class GPTConfig:
    block_size: int = 512
    vocab_size: int = 65
    n_layers: int = 6
    n_head: int = 6
    n_embd: int = 384

class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config