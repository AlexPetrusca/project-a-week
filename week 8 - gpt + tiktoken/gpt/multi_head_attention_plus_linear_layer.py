# The way to think about this is that:
#   - self-attention is the communication between the tokens
#   - then once they've gathered all the data
#   - now they need to "think" on that data individually
#       - i.e. compute in the Linear followed by Relu

import torch
import torch.nn as nn
from torch.nn import functional as F

torch.set_default_device("mps")  # use gpu
torch.manual_seed(1337)

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200
n_embd = 32
# ------------

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('res/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
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

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        # compute attention scores ("affinities")
        wei_logits = q @ k.transpose(-2, -1) * self.head_size**-0.5  # (B, T, head_size) @ (B, head_size, T)  -->  (B, T, T)
        wei_logits = wei_logits.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei_logits, dim=-1)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v  # (B, T, T) @ (B, T, head_size)  -->  (B, T, head_size)
        return out

# todo: implementing this was very simple but what does it mean and why is better than single head?
class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim=-1)  # concatenate over the channel dimension (B, T, C)

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # token information (in token embedding space)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # positional information (in position embedding space)
        self.sa_heads = MultiHeadAttention(4, n_embd // 4)  # i.e. 4 heads of 8-dimensional self-attention
        self.ffwd = FeedForward(n_embd) # additional processing
        self.lm_head = nn.Linear(n_embd, vocab_size)  # embedding space --> vocabulary space

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.sa_heads(x) # apply one head of self-attention. (B, T, C)
        x = self.ffwd(x) # (B, T, C)
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

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

# --- Training ---
# step 0: train loss 4.1701, val loss 4.1687
# step 300: train loss 2.3934, val loss 2.4069
# step 600: train loss 2.2850, val loss 2.3258
# step 900: train loss 2.2643, val loss 2.3104
# step 1200: train loss 2.2399, val loss 2.2679
# step 1500: train loss 2.2155, val loss 2.2912
# step 1800: train loss 2.2243, val loss 2.2793
# step 2100: train loss 2.2081, val loss 2.2616
# step 2400: train loss 2.1837, val loss 2.2628
# step 2700: train loss 2.1896, val loss 2.2682

# --- Generation ---
# Dy; my to lot ise fis baloves.
#
# thamed bacathert a fore?-
#
# QUEEV:
# God an ware riste af
# frour iseareoke,
# Thours thee fis.
#
# COR De; faboruro wineed him wey prowxin apentsearr forenelar isth heis shis of an wasent watelf gique whes, by you, of afould his Go Bure Lor Sich of do thels Prat.
#
# KING Re Yoblon, thas tat
# Ore ufee to en good my afart sther,
# And of olued and-thaf out cous hais mimuine,
# Kad Rove olaife,
# No:'f slame me wice napar bleo sce ef thimre bris,
# I of ableappe cro no goak,
# AlZOMERIA:
