# Copyright © 2023-2024 Apple Inc.

import math
import time
from functools import partial

import tiktoken

import dataset
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        dims: int,
        num_heads: int,
        checkpoint: bool,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dims)
        self.pe = nn.SinusoidalPositionalEncoding(dims)
        self.transformer = nn.TransformerEncoder(
            num_layers, dims, num_heads, norm_first=True, checkpoint=checkpoint
        )
        self.out_proj = nn.Linear(dims, vocab_size)

    def __call__(self, x):
        L = x.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
        x = self.embedding(x)
        x = x + self.pe(mx.arange(L))
        x = self.transformer(x, mask)
        return self.out_proj(x)


def to_samples(context_size, dataset):
    tokens = dataset.size
    window_size = context_size + 1  # include target
    samples = tokens - window_size + 1
    X = np.lib.stride_tricks.as_strided(
        dataset,
        shape=(samples, window_size),
        strides=(dataset.itemsize, dataset.itemsize),
    )
    return X[:, :-1], X[:, 1:]


def iterate_batches(batch_size, context_size, dataset):
    inputs, targets = to_samples(context_size, dataset)
    s = 0
    while True:
        if s == 0:
            # Reset permutation:
            perm = np.random.permutation(inputs.shape[0])
        ids = perm[s : s + batch_size]
        yield inputs[ids], targets[ids]
        s += batch_size
        if s >= inputs.shape[0]:
            s = 0


batch_size = 8
context_size = 1024
steps_per_eval = 250
steps_per_report = 10
steps_per_generate = 50
max_steps = 20000

learning_rate = 3e-4
lr_warmup = 200
weight_decay = 0.1

# Load vocab and dataset:
# vocab, train, valid, test = datasets.load_dataset(args.dataset)
tokenizer = tiktoken.get_encoding('gpt2')
train, val = dataset.load("../res/tinyshakespeare.txt", tokenizer)

# Initialize model:
model = TransformerLM(
    vocab_size=tokenizer.n_vocab, num_layers=12, dims=768, num_heads=12, checkpoint=True
)
model.set_dtype(mx.bfloat16)
mx.eval(model.parameters())
nparams = sum(
    x.size for k, x in tree_flatten(model.parameters()) if "embedding" not in k
)
print(f"Training a transformer with {nparams / 1024**2:.3f} M parameters")

def loss_fn(model, x, y, reduce=True):
    logits = model(x)
    losses = nn.losses.cross_entropy(logits, y)
    return mx.mean(losses) if reduce else mx.mean(losses, axis=(-1, -2))

optimizer = optim.AdamW(
    learning_rate=learning_rate, weight_decay=weight_decay
)

def eval_fn(dataset):
    inputs, targets = map(mx.array, to_samples(context_size, dataset))
    loss = 0
    for s in range(0, targets.shape[0], batch_size):
        bx, by = inputs[s : s + batch_size], targets[s : s + batch_size]
        bx, by = map(mx.array, (bx, by))
        losses = loss_fn(model, bx, by, reduce=False)
        loss += mx.sum(losses).item()
    return loss / len(targets)

state = [model.state, optimizer.state]

@partial(mx.compile, inputs=state, outputs=state)
def step(inputs, targets):
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad_fn(model, inputs, targets)
    optimizer.update(model, grads)
    return loss

train_iterator = iterate_batches(batch_size, context_size, train)
losses = []
tic = time.perf_counter()
for it, (inputs, targets) in zip(range(max_steps), train_iterator):
    model.train()
    optimizer.learning_rate = min(1, it / lr_warmup) * learning_rate
    inputs, targets = map(mx.array, (inputs, targets))
    loss = step(inputs, targets)
    mx.eval(state)
    losses.append(loss.item())

    if (it + 1) % steps_per_report == 0:
        train_loss = np.mean(losses)
        toc = time.perf_counter()
        n_tok = steps_per_report * batch_size * context_size
        print(
            f"Iter {it + 1}: Train loss {train_loss:.3f}, "
            f"Tok/sec {n_tok / (toc - tic):.3f}"
        )
        losses = []
        tic = time.perf_counter()

    # if (it + 1) % steps_per_eval == 0:
    #     model.eval()
    #     val_loss = eval_fn(val)
    #     toc = time.perf_counter()
    #     print(
    #         f"Iter {it + 1}: "
    #         f"Val loss {val_loss:.3f}, "
    #         f"Val ppl {math.exp(val_loss):.3f}, "
    #         f"Val took {(toc - tic):.3f}s, "
    #     )
    #     tic = time.perf_counter()

    if (it + 1) % steps_per_generate == 0:
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = tokenizer.encode("KING EDWARD IV:")
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

            # advance by one token
            xgen = mx.concatenate([xgen, xcol], axis=1)

        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = tokenizer.decode(tokens)
            print(f"sample {i}: {decoded}")
