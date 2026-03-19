import numpy as np

class Word2VecSGNS:
    def __init__(self, vocab_size, embedding_dim=50, learning_rate=0.01):

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate

        limit = 0.5 / self.embedding_dim

        self.W_target = np.random.uniform(-limit, limit, (self.vocab_size, self.embedding_dim))

        self.W_context = np.random.uniform(-limit, limit, (self.vocab_size, self.embedding_dim))

    def sigmoid(self, x):

        x = np.clip(x, -10, 10)
        return 1 / (1 + np.exp(-x))

    def forward_backward(self, target_idx, context_idx, negative_indices):

        pass

    def update_weights(self, grad_target, grad_context, grad_negatives, target_idx, context_idx, negative_indices):

        pass