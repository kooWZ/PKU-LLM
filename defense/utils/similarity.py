import argparse
import os
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import spacy


def get_similarity(s1, s2, method="spacy", misc=None):
    if method == "spacy":
        doc1 = misc(s1)
        doc2 = misc(s2)
        similarity_score = doc1.similarity(doc2)
    elif method == "transformer":
        # misc: [tokenizer,model]
        import torch
        from scipy.spatial.distance import cosine

        tokenizer = misc[0]
        model = misc[1]

        def get_embedding(text):
            input_ids = tokenizer(
                text, padding=True, truncation=True, return_tensors="pt"
            )["input_ids"]

            with torch.no_grad():
                outputs = model(input_ids)

            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze()

            return embeddings

        embedding1 = get_embedding(s1)
        embedding2 = get_embedding(s2)
        similarity_score = 1 - cosine(embedding1, embedding2)
    return similarity_score


def get_divergence(similarity_matrix, i, j):
    p = similarity_matrix[i] / np.sum(similarity_matrix[i])
    q = similarity_matrix[j] / np.sum(similarity_matrix[j])
    divergence = np.sum(p * np.log(p / q))
    return divergence


def visualize(divergence_matrix, save_path, vmax):
    plt.figure(figsize=(8, 8))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
    plt.imshow(divergence_matrix, cmap="viridis", interpolation="nearest", norm=norm)
    plt.colorbar(label="Divergence")

    plt.title("Divergence Matrix Heatmap")
    plt.xticks(range(divergence_matrix.shape[0]))
    plt.yticks(range(divergence_matrix.shape[0]))

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
