import os

import numpy as np
import tiktoken


def generate_transformer_dataset(file_path, train_split_ratio=0.9, sequence_length=1024):
    # Ensure the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Read the text file
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Initialize tokenizer (using cl100k_base which works for GPT-3.5 and GPT-4)
    tokenizer = tiktoken.get_encoding('gpt2')

    # Tokenize the entire text
    tokens = tokenizer.encode(text)

    # Calculate split indices
    total_tokens = len(tokens)
    train_tokens_count = int(total_tokens * train_split_ratio)

    # Create sequences of fixed length
    def create_sequences(token_list, seq_length):
        sequences = []
        for i in range(0, len(token_list) - seq_length, seq_length):
            sequence = token_list[i:i + seq_length]
            sequences.append(sequence)
        return np.array(sequences)

    # Split tokens into train and validation
    train_tokens = create_sequences(tokens[:train_tokens_count], sequence_length)
    val_tokens = create_sequences(tokens[train_tokens_count:], sequence_length)

    return train_tokens, val_tokens