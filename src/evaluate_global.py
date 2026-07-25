import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import insightface

from trainer import FineTuner

# ----------------------------------------
# Load Global Federated Model
# ----------------------------------------

global_model = FineTuner()

global_model.load_state_dict(
    torch.load("outputs/global_model.pt", map_location="cpu")
)

global_model.eval()

print("Global federated model loaded.")

# ----------------------------------------
# Load InsightFace
# ----------------------------------------

face_model = insightface.app.FaceAnalysis(
    name="buffalo_l"
)

face_model.prepare(ctx_id=0)

print("InsightFace loaded.")

# ----------------------------------------
# Get Original 512-D Embedding
# ----------------------------------------

def get_original_embedding(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return None

    faces = face_model.get(image)

    if len(faces) == 0:
        return None

    embedding = torch.tensor(
        faces[0].embedding,
        dtype=torch.float32,
    )

    return embedding

# ----------------------------------------
# Get Federated Embedding
# ----------------------------------------

def get_global_embedding(image_path):

    original_embedding = get_original_embedding(
        image_path
    )

    if original_embedding is None:
        return None

    with torch.no_grad():

        new_embedding = global_model(
            original_embedding.unsqueeze(0)
        )

    return new_embedding.squeeze(0).numpy()
# ----------------------------------------
# Load Embeddings From Evaluation Dataset
# ----------------------------------------

def load_embeddings(eval_folder):

    embeddings = {}

    print("\nExtracting embeddings...")

    for person in sorted(os.listdir(eval_folder)):

        person_folder = os.path.join(
            eval_folder,
            person
        )

        if not os.path.isdir(person_folder):
            continue

        person_embeddings = []

        for image in os.listdir(person_folder):

            image_path = os.path.join(
                person_folder,
                image
            )

            emb = get_global_embedding(image_path)

            if emb is not None:
                person_embeddings.append(emb)

        if len(person_embeddings) > 0:

            embeddings[person] = person_embeddings

            print(
                f"{person}: {len(person_embeddings)} embeddings"
            )

    return embeddings
# ----------------------------------------
# Generate Positive and Negative Pairs
# ----------------------------------------

def generate_pairs(embeddings):

    X = []
    y = []

    people = list(embeddings.keys())

    print("\nGenerating positive pairs...")

    for person in people:

        embs = embeddings[person]

        for i in range(len(embs)):

            for j in range(i + 1, len(embs)):

                X.append((embs[i], embs[j]))
                y.append(1)

    print("Generating negative pairs...")

    for i in range(len(people)):

        for j in range(i + 1, len(people)):

            for emb1 in embeddings[people[i]]:

                for emb2 in embeddings[people[j]]:

                    X.append((emb1, emb2))
                    y.append(0)

    print(f"\nTotal Evaluation Pairs: {len(X)}")

    return X, y
# ----------------------------------------
# Compute Cosine Similarities
# ----------------------------------------

def compute_similarities(pairs):

    print("\nCalculating similarities...")

    similarities = []

    for emb1, emb2 in tqdm(pairs):

        sim = cosine_similarity(
            [emb1],
            [emb2]
        )[0][0]

        similarities.append(sim)

    return np.array(similarities)

# ----------------------------------------
# Evaluate Federated Model
# ----------------------------------------

def evaluate_model(eval_folder):

    embeddings = load_embeddings(eval_folder)

    pairs, labels = generate_pairs(embeddings)

    similarities = compute_similarities(pairs)

    # ----------------------------------------
    # Find Best Threshold
    # ----------------------------------------

    thresholds = np.arange(0.0, 1.01, 0.01)

    best_accuracy = 0
    best_threshold = 0.5

    for threshold in thresholds:

        predictions = (similarities >= threshold).astype(int)

        acc = accuracy_score(labels, predictions)

        if acc > best_accuracy:
            best_accuracy = acc
            best_threshold = threshold

    print(f"\nBest Threshold : {best_threshold:.2f}")
    print(f"Best Accuracy  : {best_accuracy:.4f}")

    predictions = (similarities >= best_threshold).astype(int)

    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    fpr, tpr, _ = roc_curve(labels, similarities)
    roc_auc = auc(fpr, tpr)

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"AUC       : {roc_auc:.4f}")

    return (
        similarities,
        labels,
        best_threshold,
        best_accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        fpr,
        tpr,
    )
# ----------------------------------------
# Similarity Distribution
# ----------------------------------------

def plot_similarity(similarities, labels):

    positive = [
        s for s, l in zip(similarities, labels)
        if l == 1
    ]

    negative = [
        s for s, l in zip(similarities, labels)
        if l == 0
    ]

    plt.figure(figsize=(8,5))

    plt.hist(
        positive,
        bins=50,
        alpha=0.5,
        label="Same Person"
    )

    plt.hist(
        negative,
        bins=50,
        alpha=0.5,
        label="Different Person"
    )

    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title("Federated Similarity Distribution")

    plt.legend()

    plt.savefig(
        "outputs/similarity_distribution_global.png"
    )

    plt.close()
# ----------------------------------------
# ROC Curve
# ----------------------------------------

def plot_roc(fpr, tpr, roc_auc):

        plt.figure(figsize=(6,6))

        plt.plot(
            fpr,
            tpr,
            label=f"AUC = {roc_auc:.4f}"
        )

        plt.plot(
            [0,1],
            [0,1],
            linestyle="--"
        )

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")

        plt.title("Federated ROC Curve")

        plt.legend()

        plt.savefig(
            "outputs/roc_curve_global.png"
        )

        plt.close()

    # ----------------------------------------
# Precision Recall Curve
# ----------------------------------------

def plot_pr(similarities, labels):

        precision, recall, _ = precision_recall_curve(
            labels,
            similarities
        )

        plt.figure(figsize=(6,6))

        plt.plot(
            recall,
            precision
        )

        plt.xlabel("Recall")
        plt.ylabel("Precision")

        plt.title("Federated Precision Recall Curve")

        plt.savefig(
            "outputs/precision_recall_global.png"
        )

        plt.close()

# ----------------------------------------
# Save Evaluation Metrics
# ----------------------------------------

def save_metrics(
    threshold,
    accuracy,
    precision,
    recall,
    f1,
    roc_auc,
):

    os.makedirs("outputs", exist_ok=True)

    with open(
        "outputs/global_metrics.txt",
        "w",
    ) as file:

        file.write("Federated Face Recognition Results\n")
        file.write("=" * 40 + "\n\n")

        file.write(f"Best Threshold : {threshold:.4f}\n")
        file.write(f"Accuracy       : {accuracy:.4f}\n")
        file.write(f"Precision      : {precision:.4f}\n")
        file.write(f"Recall         : {recall:.4f}\n")
        file.write(f"F1 Score       : {f1:.4f}\n")
        file.write(f"AUC            : {roc_auc:.4f}\n")

    print("\nMetrics saved to outputs/global_metrics.txt")

# ----------------------------------------
# Main
# ----------------------------------------

if __name__ == "__main__":

    EVAL_FOLDER = "dataset/eval"      # <-- CHANGE THIS IF NEEDED

    (
        similarities,
        labels,
        threshold,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        fpr,
        tpr,
    ) = evaluate_model(EVAL_FOLDER)

    plot_similarity(
        similarities,
        labels,
    )

    plot_roc(
        fpr,
        tpr,
        roc_auc,
    )

    plot_pr(
        similarities,
        labels,
    )

    save_metrics(
        threshold,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
    )

    print("\nEvaluation completed successfully.")
