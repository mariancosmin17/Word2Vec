import numpy as np
from dataset import Word2VecDataset
from model import Word2VecSGNS

def train_word2vec(text, embedding_dim=10, window_size=2, num_epochs=50, learning_rate=0.05, num_neg_samples=5):
    print("1. Pregătirea datelor...")
    dataset = Word2VecDataset(text, window_size=window_size)
    print(f"Vocabular: {dataset.vocab_size} cuvinte unice.")

    training_pairs = dataset.get_training_pairs()
    print(f"Perechi de antrenament: {len(training_pairs)}")

    print("\n2. Inițializarea modelului...")
    model = Word2VecSGNS(dataset.vocab_size, embedding_dim, learning_rate)

    print("\n3. Începerea antrenamentului...")
    for epoch in range(num_epochs):
        total_loss = 0

        np.random.shuffle(training_pairs)

        for target_idx, context_idx in training_pairs:
            negative_indices = dataset.get_negative_samples(num_neg_samples)

            loss, grad_target, grad_context, grad_neg = model.forward_backward(
                target_idx, context_idx, negative_indices
            )

            total_loss += loss

            model.update_weights(
                target_idx, context_idx, negative_indices,
                grad_target, grad_context, grad_neg
            )

        avg_loss = total_loss / len(training_pairs)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoca {epoch + 1:2d}/{num_epochs} | Loss mediu: {avg_loss:.4f}")

    print("\nAntrenament finalizat!")
    return dataset, model


if __name__ == "__main__":
    sample_text = """
    Machine learning is fascinating. Deep learning models like word2vec 
    capture semantic meaning from text. Natural language processing relies 
    on these continuous vector representations. Vector representations are 
    fundamental to modern artificial intelligence.
    """

    print("=== Pornire Word2Vec Numpy ===")
    dataset, trained_model = train_word2vec(
        text=sample_text,
        embedding_dim=10,
        window_size=2,
        num_epochs=50,
        learning_rate=0.05,
        num_neg_samples=5
    )