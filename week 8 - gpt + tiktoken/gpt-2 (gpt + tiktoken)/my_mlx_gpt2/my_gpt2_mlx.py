import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from functools import partial

import tiktoken
import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.nn.losses as F
import mlx.optimizers as optim
from mlx.utils import tree_flatten

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.n_heads = config.n_head
        self.n_embd = config.n_embd
        self.causal_mask = CausalSelfAttention.create_additive_causal_mask(config.block_size, dtype=config.dtype)

        self.query_proj = nn.Linear(self.n_embd, self.n_embd)
        self.key_proj = nn.Linear(self.n_embd, self.n_embd)
        self.value_proj = nn.Linear(self.n_embd, self.n_embd)
        self.out_proj = nn.Linear(self.n_embd, self.n_embd)

    def __call__(self, x):
        B, T, C = x.shape
        # calculate query, key, value for all heads
        q = self.query_proj(x) # (B, T, C) -> (B, T, C)
        k = self.key_proj(x) # (B, T, C) -> (B, T, C)
        v = self.value_proj(x) # (B, T, C) -> (B, T, C)

        # reshape query, key, value to batch over n_batches x n_heads
        #   - this way we can compute attention for all heads at once (i.e. multi-head attention) with a single matrix multiply
        #   - nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        q = mx.unflatten(q, -1, (self.n_heads, -1)).transpose(0, 2, 1, 3) # (B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)
        k = mx.unflatten(k, -1, (self.n_heads, -1)).transpose(0, 2, 1, 3) # (B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)
        v = mx.unflatten(v, -1, (self.n_heads, -1)).transpose(0, 2, 1, 3) # (B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)

        # causal flash attention
        scale = math.sqrt(1 / q.shape[-1])
        output = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=self.causal_mask[:T, :T]) # 3x(B, nh, T, hs) -> (B, nh, T, hs)

        # re-assemble all head outputs side by side and project out
        output = output.transpose(0, 2, 1, 3).flatten(-2, -1) # (B, nh, T, hs) -> (B, T, nh, hs) -> (B, T, C)
        return self.out_proj(output) # (B, T, C) -> (B, T, C)

    @staticmethod
    def create_additive_causal_mask(N: int, dtype = mx.float32):
        indices = mx.arange(N)
        mask = indices[:, None] < indices[None]
        mask = mask.astype(dtype) * mx.finfo(dtype).min
        return mask

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def __call__(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def __call__(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 byte tokes + 1<|endoftext|>
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    dtype = mx.bfloat16
    # NOTE: head_size = n_embd / n_head = 64  # embedding dimension of each attention head

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = [Block(config) for _ in range(config.n_layer)],
            ln_f = nn.LayerNorm(config.n_embd),
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme (refer to [1] in play.ipynb)
        self.transformer['wte'].weight = self.lm_head.weight

    def __call__(self, idx):
        # idx is of shape (B, T)
        B, T = idx.shape
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and position embeddings
        pos = mx.arange(0, T, dtype=mx.int32)  # shape (T)
        pos_emb = self.transformer['wpe'](pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer['wte'](idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd) + (T, n_embd) -> (B, T, n_embd)
        # forward the blocks of the transformer
        for block in self.transformer['h']:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer['ln_f'](x)
        return self.lm_head(x)  # (B, T, vocab_size)

#-----------------------------------------------------------------------------------------------------------------------

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = mx.array(npt, dtype=mx.uint32)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, split):
        self.B = B
        self.T = T
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "../res/edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).reshape(B, T) # inputs
        y = (buf[1:]).reshape(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = 0
        return x, y

#-----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    gpt_config = GPTConfig()

    enc = tiktoken.get_encoding("gpt2")

    # data loaders
    B = 8  # micro batch size
    T = 1024  # sequence length

    train_loader = DataLoaderLite(B=B, T=T, split="train")
    val_loader = DataLoaderLite(B=B, T=T, split="val")

    # model
    model = GPT(gpt_config)
    model.set_dtype(gpt_config.dtype)
    mx.eval(model.parameters())
    nparams = sum(x.size for k, x in tree_flatten(model.parameters()) if "embedding" not in k)
    print(f"Training a transformer with {nparams / 1024**2:.3f} M parameters")

    # optimizer
    optimizer = optim.AdamW(learning_rate=3e-4, betas=[0.9, 0.95], eps=1e-8, weight_decay=0.1)

    def loss_fn(model, x, y, reduce=True):
        logits = model(x)
        losses = F.cross_entropy(logits, y)
        return mx.mean(losses) if reduce else mx.mean(losses, axis=(-1, -2))

    state = [model.state, optimizer.state]
    @partial(mx.compile, inputs=state, outputs=state)
    def optimizer_step(inputs, targets):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, inputs, targets)
        optimizer.update(model, grads)
        return loss

    def generate_step(step):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = mx.array(tokens, dtype=mx.int32)  # (8 tokens,)
        xgen = mx.repeat(mx.expand_dims(tokens, axis=0), num_return_sequences, axis=0)  # (5 rows, 8 tokens)
        while xgen.shape[1] < max_length:
            # forward the model to get the logits
            logits = model(xgen)  # (B, T, vocab_size)
            # take the logits at the last position
            logits = logits[:, -1, :]  # (B, vocab_size)
            # get the probabilities
            probs = nn.softmax(logits, axis=-1)

            # get the top k probabilities
            k = 50
            topk_indices = mx.argsort(logits, axis=-1)[:, -k:]
            topk_logits = mx.sort(logits, axis=-1)[:, -k:]

            # select a token from the top probabilities
            ix = mx.random.categorical(topk_logits, num_samples=1)  # (B, 1)
            xcol = mx.take_along_axis(topk_indices, indices=ix, axis=-1)

            xgen = mx.concatenate([xgen, xcol], axis=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"sample {i}: {decoded}")

    def validation_step(step):
        model.eval()
        val_loader.reset()
        val_loss_accum = 0.0
        val_loss_steps = 20
        for _ in range(val_loss_steps):
            x, y = val_loader.next_batch()
            loss = loss_fn(model, x, y) / val_loss_steps
            val_loss_accum += loss

        print(f"validation loss: {val_loss_accum.item():.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} val {val_loss_accum.item():.4f}\n")

    max_steps = 19073  # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
    def train_step(step):
        t0 = time.time()
        last_step = (step == max_steps - 1)

        # once in a while evaluate our validation loss
        if step % 250 == 0 or last_step:
            validation_step(step)

        # once in a while generate from the model (except step 0, which is noise)
        if (step > 0 and step % 50 == 0) or last_step:
            generate_step(step)

        # do one step of the optimization
        model.train()
        x, y = train_loader.next_batch()
        loss = optimizer_step(x, y)
        mx.eval(state)

        t1 = time.time()
        dt = t1 - t0  # time difference in seconds
        tokens_processed = train_loader.B * train_loader.T
        tokens_per_sec = tokens_processed / dt
        print(
            f"step {step:5d} | loss: {loss:.6f} | dt: {dt * 1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss:.6f}\n")

    # create the log directory we will write checkpoints to and log to
    log_dir = "my_mlx_impl/log"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log.txt")
    with open(log_file, "w") as f:  # open for writing to clear the file
        pass

    # train
    start = datetime.now()
    num_epochs = 1000
    for i in range(num_epochs):
        train_step(i)
    end = datetime.now()
    print(f"total time: {end - start}")
    print(f"average tokens/sec: {(train_loader.B * train_loader.T * num_epochs) / (end.timestamp() - start.timestamp())}")
