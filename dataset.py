import numpy as np
import re
from collections import Counter


class Word2VecDataset:
    def __init__(self, text, window_size=2):
        self.window_size = window_size

        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        self.words = text.split()

        word_counts = Counter(self.words)
        self.vocab = list(word_counts.keys())
        self.vocab_size = len(self.vocab)

        self.word2idx = {word: i for i, word in enumerate(self.vocab)}
        self.idx2word = {i: word for i, word in enumerate(self.vocab)}

        self.data_indices = [self.word2idx[w] for w in self.words]

        counts = np.array([word_counts[word] for word in self.vocab])
        freqs = counts ** 0.75
        self.word_probs = freqs / np.sum(freqs)

    def get_training_pairs(self):

        pairs = []
        for i, target_idx in enumerate(self.data_indices):

            start = max(0, i - self.window_size)
            end = min(len(self.data_indices), i + self.window_size + 1)

            for j in range(start, end):
                if i != j:
                    context_idx = self.data_indices[j]
                    pairs.append((target_idx, context_idx))

        return pairs

    def get_negative_samples(self, num_samples):

        return np.random.choice(self.vocab_size, size=num_samples, p=self.word_probs)