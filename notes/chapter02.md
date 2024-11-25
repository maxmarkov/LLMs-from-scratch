# Chapter 02. Working with text data

## 1. Text to tokens

Tokens are individual words or special characters, including punctuation. To build a vocabulary, use a large and diverse training set, especially for developing large language models (LLMs), to ensure extensive vocabulary coverage.

## 2. Tokens to token IDs

Convert text tokens into integer representations (token IDs) by mapping each unique token to a unique integer ID from the vocabulary. This assigns a distinct ID to each token, facilitating efficient processing. Use inverse vocabulary to convert token IDs back to tokens

## Unknown Words and Special Tokens

To manage unknown words, special tokens are used to enhance a model’s contextual understanding:

- `<|unk|>`: represents unknown words
- `<|endoftext|>`: marks the end of text
- `<|pad|>:` used for padding to ensure consistent text length

## Byte Pair Encoding (BPE)

GPT models use BPE tokenization. BPE handles unknown words by breaking them into smaller subwords or individual characters. It merges frequently occurring characters into subwords and frequent subwords into whole words, enabling it to deal with out-of-vocabulary words.

## 3. Self-labeling with input-target pairs

Generate input-target pairs for training using a sliding window approach. The sliding window extracts subsequences of a predefined length from the text and their corresponding targets. The targets are subsequences shifted by a stride relative to the inputs. If the stride equals the predefined length, there is no overlap with the input. Implement a DataLoader to fetch these pairs for model training.

## 4. Convert Token IDs into Embedding Vectors

Initialize the embedding weights randomly. Use the vocabulary size and the desired embedding dimension to create an embedding layer with PyTorch using `torch.nn.Embedding`. The embedding weights will be optimized during training. This process is equivalent to one-hot encoding followed by matrix multiplication in a fully connected layer to produce dense vector representations for each token.

## 5. Encode positional information

So far, each token ID is always mapped to the same vector representation. To enhance an LLM’s ability to understand token order and relationships, position-aware embeddings are used:

- **Absolute positional embeddings** are directly associated with specific positions in a sequence. A unique positional embedding is added to each token embedding to convey its exact location in the sequence.
- **Relative positional embeddings** focus on the relative position or distance between tokens, capturing relationships based on proximity rather than fixed positions.

## Other
Smaller batches require less memory during training but lead to more noisy model updates

[Chapter 02 illustration](./chapter02-1.png)