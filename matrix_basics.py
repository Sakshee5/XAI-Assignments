import torch

"""1. Embedding Matrix
Lets assume an embedding matrix E has dimensions [V x D], where:

V is the size of the vocabulary (number of tokens).
D is the dimension of each embedding vector.

For example, if V = 3 (3 tokens in the vocabulary) and D = 4 (4-dimensional embeddings), then E would look like:"""

E = torch.tensor([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.9, 1.0, 1.1, 1.2]
])

"""One-Hot Vector
A one-hot encoded vector for a single token is a row vector of size [1 x V]. 

For example, to encode Token 1, the one-hot vector would be:
"""
one_hot_1 = torch.tensor([0, 1, 0], dtype=torch.float32)

# Multiplying [1 x V] by [V x D] results in [1 x D], giving you the embedding vector for the token.
embedding_1 = torch.matmul(one_hot_1, E)

print("Embedding Matrix:")
print(E)
print("\nOne-hot Vector for Token 1:")
print(one_hot_1)
print("\nEmbedding for Token 1:")
print(embedding_1)
