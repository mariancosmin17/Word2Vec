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

        v_target = self.W_target[target_idx]
        v_context = self.W_context[context_idx]
        v_negatives = self.W_context[negative_indices]

        score_pos = np.dot(v_target, v_context)
        pred_pos = self.sigmoid(score_pos)

        scores_neg = np.dot(v_negatives, v_target)
        preds_neg = self.sigmoid(scores_neg)

        loss_pos = -np.log(pred_pos + 1e-10)
        loss_neg = -np.sum(np.log(1 - preds_neg + 1e-10))
        total_loss = loss_pos + loss_neg

        error_pos = pred_pos - 1.0

        errors_neg = preds_neg

        grad_target = np.zeros(self.embedding_dim)

        grad_target += error_pos * v_context
        grad_context = error_pos * v_target

        grad_target += np.dot(errors_neg, v_negatives)

        grad_negatives = np.outer(errors_neg, v_target)

        return total_loss, grad_target, grad_context, grad_negatives

    def update_weights(self, target_idx, context_idx, negative_indices,
                       grad_target, grad_context, grad_negatives):

        self.W_target[target_idx] -= self.learning_rate * grad_target
        self.W_context[context_idx] -= self.learning_rate * grad_context

        for i, neg_idx in enumerate(negative_indices):
            self.W_context[neg_idx] -= self.learning_rate * grad_negatives[i]