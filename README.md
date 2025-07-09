# # Book Genre ClassifierWith Neural Network (from Scratch in NumPy)

This project builds a neural network from scratch (using only NumPy) to classify the genre of books based on their descriptions. It uses the Goodreads Best Books dataset.

## Dataset

- **Source:** [Kaggle - Goodreads Best Books](https://www.kaggle.com/datasets/meetnaren/goodreads-best-books)
- **File:** `book_data.csv`
- **Columns used:**
  - `book_desc`: Description of the book
  - `genres`: List of genres (split by `|`)

## Overview

The goal is to predict the **primary genre** of a book from its description. The process includes:

- Downloading and preprocessing the dataset
- Extracting text features with TF-IDF
- Dimensionality reduction using PCA
- Training a fully connected neural network (with two hidden layers) from scratch using NumPy
- Evaluating the classifier

---

## Features

- Text preprocessing with `scikit-learn`’s `TfidfVectorizer` (5000 features)
- Label encoding of top 15 genres
- Dimensionality reduction to ~3850 features with PCA (95% variance)
- 3-layer neural network with ReLU activations
- Cross-entropy loss with L2 regularization
- Adam optimizer implemented manually
- Loss curve visualization
- Performance evaluation with accuracy, classification report, and confusion matrix
---

## Training

- **Batch size:** 64
- **Epochs:** 100
