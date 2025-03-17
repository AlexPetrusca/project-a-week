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
learning_rate = 1e-3 # reduce learning rate
eval_iters = 200
n_embd = 32
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

class TransformerBlock(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)   # feed forward per token (cuz applies only to last dimension)

    def forward(self, x):
        x = self.sa(x)
        x = self.ffwd(x)
        return x

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # token information (in token embedding space)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # positional information (in position embedding space)
        self.blocks = nn.Sequential(
            TransformerBlock(n_embd, n_head=4),
            TransformerBlock(n_embd, n_head=4),
            TransformerBlock(n_embd, n_head=4),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)  # embedding space --> vocabulary space

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        x = self.blocks(x) # (B, T, C)
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
# step 0: train loss 4.1824, val loss 4.1804
# step 300: train loss 3.1002, val loss 3.1144
# step 600: train loss 2.8865, val loss 2.8885
# step 900: train loss 2.7758, val loss 2.7663
# step 1200: train loss 2.6832, val loss 2.6759
# step 1500: train loss 2.5962, val loss 2.5959
# step 1800: train loss 2.5483, val loss 2.5352
# step 2100: train loss 2.4940, val loss 2.4933
# step 2400: train loss 2.4669, val loss 2.4750
# step 2700: train loss 2.4518, val loss 2.4577

# --- Generation ---
# Se gtheded bacarhs,
# Ga fe ghet're
# the hit fourl mrricre If
# EUXN:
# Rherer met
# Thew ses ce fis.
#
#  an fat faeceuro wine bo pila ye powe I ape to,
# ar fomenelnl Tith hrie sed bombou whount If she gint to mise yo wid ony
# bomlled fir pedal wow thar o hiudt sysan woo Roo-r'le to ton, thhe tas nrilufe at;
# en gin I jpacargf cher,
# Ann inqolund af stonf' corceror.
#
# Wiopmriaeorg dow he ot:
# An, thhof fot eid band grapar bieo sce; for pore boic,
# I o tar kappe cel nas wak o lic fouge so On garemr.
#
# MEIICCLOUTESN