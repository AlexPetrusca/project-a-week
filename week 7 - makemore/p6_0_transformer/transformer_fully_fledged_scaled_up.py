# The way to think about this is that:
#   - self-attention is the communication between the tokens
#   - then once they've gathered all the data
#   - now they need to "think" on that data individually
#       - i.e. compute in the Linear followed by Relu

# We can stack these layers now:
#   - attention block 1
#       - MultiHeadAttention 1
#       - Linear 1
#       - Relu 1
#   - attention block 2
#       - MultiHeadAttention 2
#       - Linear 2
#       - Relu 2
#   - ...

# communication/computation sandwiches:
#   - communication -> computation -> communication -> computation -> ...

# This network ends up getting pretty deep, hence "deep learning".
# Without residual connections and normalization layers, the network is unstable and hard to train.
# Just adding the residual connections stabilizes the network (and it performs way better... wtf!).
# Slightly better stability with layer norm
# todo: add comment on dropout
# todo: add comment on hyperparameters

import torch
import torch.nn as nn
from torch.nn import functional as F

torch.set_default_device("mps")  # use gpu
torch.manual_seed(1337)

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_head = 6  # every head is 64 dimensional
n_transformer_blocks = 6
dropout = 0.2
# ------------

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('../res/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    m.train()
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        # compute attention scores ("affinities")
        wei_logits = q @ k.transpose(-2, -1) * self.head_size**-0.5  # (B, T, head_size) @ (B, head_size, T)  -->  (B, T, T)
        wei_logits = wei_logits.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei_logits, dim=-1)
        wei = self.dropout(wei)  # prevent some of the nodes from communicating with dropout (avoids overfitting) (creates ensemble)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, T) @ (B, T, head_size)  -->  (B, T, head_size)
        return out

# todo: implementing this was very simple but what does it mean and why is better than single head?
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])  # heads of attention
        self.proj = nn.Linear(n_embd, n_embd)  # "projection back into the residual pathway"
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # concatenate over the channel dimension (B, T, C)
        out = self.proj(out)  # "project back into the residual pathway"
        out = self.dropout(out)   # apply droupout on residual path
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
        )
        self.proj = nn.Linear(4 * n_embd, n_embd)  # "project back into the residual pathway"
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.net(x)
        out = self.proj(out)  # "project back into the residual pathway"
        out = self.dropout(out)  # apply droupout on residual path
        return out

class TransformerBlock(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)  # feed forward per token (cuz applies only to last dimension)
        self.ln1 = nn.LayerNorm(n_embd)  # batch norm per token (cuz applies only to last dimension)
        self.ln2 = nn.LayerNorm(n_embd)  # batch norm per token (cuz applies only to last dimension)

    def forward(self, x):
        # `x = x + <some computation>` is the residual pathway...
        x = x + self.sa(self.ln1(x))  # MultiHeadAttention now also "projects back into the residual pathway"
        x = x + self.ffwd(self.ln2(x))  # FeedForward now also "projects back into the residual pathway"
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # token information (in token embedding space)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # positional information (in position embedding space)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head) for _ in range(n_transformer_blocks)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)  # embedding space --> vocabulary space

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx (B, T) array of indices in the current context
            # (never pass more than block_size tokens)
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

m = BigramLanguageModel()

# create a PyTorch optimizer
torch.set_default_device("cpu")  # use cpu
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
torch.set_default_device("mps")  # back to gpu

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

losses = estimate_loss()
print(f"Final: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

# --- Training ---
# step 0: train loss 4.3283, val loss 4.3285
# step 500: train loss 1.9113, val loss 2.0072
# step 1000: train loss 1.5471, val loss 1.7220
# step 1500: train loss 1.4018, val loss 1.6085
# step 2000: train loss 1.3191, val loss 1.5506
# step 2500: train loss 1.2571, val loss 1.5096
# step 3000: train loss 1.2121, val loss 1.5016
# step 3500: train loss 1.1690, val loss 1.4926
# step 4000: train loss 1.1311, val loss 1.4867
# step 4500: train loss 1.0956, val loss 1.4887
# Final: train loss 1.0606, val loss 1.4919

# --- Generation ---
# Paged be in affe we both grows it.
# Almost with a petite of this plant Thursday with the
# slanders; when it 'twill the prodigress.
#
# FRIAR LAURENCE:
# From the cold tumond of that e'er set
# For brother.
#
# CARDISS OF:
# A though I people
# Tyrs' to me: 'tis good sons received, pity and that
# prosecures of her deed own crepting: 'tis her, his exercise:
# Let his sorrow go we her.
#
# FRIAR LAURENCY:
# Return, brother of England's Sly, Den.
# The manst thou eopen, sit in the hole day,
# The dead constantly, but I say, or