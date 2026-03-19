# Word2Vec from Scratch (Pure NumPy)

This repository contains a pure NumPy implementation of the Word2Vec algorithm, specifically the **Skip-Gram with Negative Sampling (SGNS)** architecture. It demonstrates the core mechanics of representation learning, including data preprocessing, forward pass operations, manual gradient derivations (backpropagation), and vectorized inference, without relying on deep learning frameworks like PyTorch or TensorFlow.

## Design Decisions & Trade-offs

### 1. Data Preprocessing
* **Object-Oriented Encapsulation:** The vocabulary building, tokenization, and index mappings (`word2idx`, `idx2word`) are encapsulated in a `Word2VecDataset` class to cleanly separate data logic from the neural network math.
* **Negative Sampling Distribution:** Implemented the smoothed unigram distribution trick from Mikolov et al. (frequencies raised to the 3/4ths power) to generate negative samples. This prevents highly frequent stop-words from overwhelmingly dominating the negative samples.

### 2. Model Architecture & Initialization
* **Dual Weight Matrices:** The architecture maintains separate embedding matrices for target words (`W_target`) and context words (`W_context`). This decoupling simplifies the gradient derivations and leads to a more stable optimization landscape compared to sharing a single matrix.
* **Weight Initialization:** Weights are initialized using a uniform distribution scaled by the embedding dimension `[-0.5/D, 0.5/D]`. This prevents vanishing/exploding gradients in the early stages of training.
* **Numerical Stability:** The sigmoid activation function includes an `np.clip` safeguard to prevent exponential overflow during forward passes with large dot products, ensuring `NaN` values do not corrupt the weight matrices.

### 3. Mathematical Optimization
* **Vectorized Operations:** In the `forward_backward` pass, calculating the loss and gradients for the `K` negative samples is fully vectorized using `np.dot` and `np.outer` instead of Python `for` loops. This leverages underlying C/BLAS libraries via NumPy, drastically speeding up the gradient computation.
* **Log Stability:** Added a small epsilon (`1e-10`) to the `-np.log()` function to prevent `NaN` values caused by computing the logarithm of absolute zero when predictions are completely wrong or right.
* **Gradient Accumulation:** For negative samples, the gradient with respect to the target vector is a sum of the errors scaled by the respective negative context vectors. This allows us to compute the target update in a mathematically correct, single pass.

### 4. Training Loop & Optimization
* **Stochastic Gradient Descent (SGD):** The training loop uses a standard SGD approach. Crucially, the training pairs are shuffled via `np.random.shuffle` at the beginning of each epoch. This breaks the sequential correlation of text data, reducing variance in the gradient updates and preventing the model from getting stuck in local minima.
* **Loss Monitoring:** The loop calculates the average loss per epoch. A steadily decreasing binary cross-entropy loss verifies that the pure NumPy gradient derivations are mathematically correct and the vectors are successfully capturing context representations.

### 5. Inference & Evaluation
* **Cosine Similarity Matrix Operations:** Finding the most similar words requires computing the cosine similarity between the target vector and the entire vocabulary. This is implemented via a fully vectorized matrix-vector multiplication, which evaluates all vocabulary candidates instantly.
* **Embedding Extraction:** Following the original C implementation of Word2Vec, the final semantic representations are drawn exclusively from the target weight matrix (`W_target`).
