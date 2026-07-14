This repository contains an end-to-end NLP pipeline designed to classify customer service queries into distinct banking intents, such as card activation, payment failures, refund tracking. 
I benchmarked a traditional bag-of-words baseline against transformer architecture to evaluate the impact of contextual bidirectional embeddings on semantically overlapping financial classes.

## Performance at a Glance

The evaluation was performed on a perfectly balanced test dataset containing exactly 40 evaluation samples per intent class (3,080 total test rows). As a result of this perfect class distribution, the **Macro F1** and **Weighted F1** metrics were mathematically identical, providing an unbiased look at true model capability.

| Evaluation Metric | Baseline MLP (TF-IDF + Bigrams) | Fine-Tuned BERT (Transformer) | Absolute Improvement |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | 84.94% | **89.16%** | **+4.22%** |
| **Macro F1-Score**   | 0.8501 | **0.8873** | **+0.0372** |
| **Weighted F1-Score**| 0.8501 | **0.8873** | **+0.0372** |

## Project Architecture

The project is structured as two independent machine learning pipelines:

### Pipeline A: The Baseline MLP

*   **Vectorisation:** TF-IDF representation with explicit bigram constraints to capture compound financial terms (e.g., “top up”, “apple pay”).
*   **Classifier:** A PyTorch-based Multi-Layer Perceptron (MLP) trained with Cross-Entropy Loss.
*   **Limitations:** This model treated documents as an unordered “checklist” of words. This resulted in semantic confusion between certain classes sharing high vocabulary overlap.

### Pipeline B: Fine-Tuned BERT 

*   **Tokeniser:** Hugging Face `bert-base-uncased` with dynamic truncation and padding set to a safe `max_length=64` (optimised as a result of my exploratory data analysis which showed a maximum sequence length of ~53 words).
*   **Model:** `AutoModelForSequenceClassification` mapping contextual embeddings to 77 target classes.
*   **Training Details:** Optimised using `AdamW` with a small learning rate of `2e-5` over 3 epochs to preserve pre-trained weights and avoid catastrophic forgetting.

## Core Insights & Diagnostic Analysis

Traditional Bag-of-Words models fail when two entirely different user problems use identical terminology. BERT's bidirectional attention mechanisms successfully solved these key points of failure. For example:

*   **`card_acceptance` vs. `card_not_working`**
    *   **MLP Performance:** F1-Score: `0.73`
    *   **BERT Performance:** F1-Score: `0.83` (+10% Gain)
    *   The MLP could not distinguish between "the terminal didn't accept my card" and "my card is physically not working". BERT's attention window processed the syntax and word ordering contextually to separate them.
*   BERT achieved a perfect `1.00` Precision on critical user intents like `cancel_transfer`, `beneficiary_not_allowed`, and `card_swallowed`. In production, this prevents costly errors like triggering automated card cancellations due to false positives.

## How to Run the Project

### 1. Installation
Clone this repository and install the required packages:

pip install torch transformers scikit-learn matplotlib numpy tqdm
