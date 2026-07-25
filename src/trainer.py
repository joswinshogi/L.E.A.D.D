import os
import cv2
import random
import torch
import insightface

import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

# ----------------------------------------
# Device Selection
# ----------------------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Using device: {DEVICE}")

# ----------------------------------------
# Load InsightFace Model
# ----------------------------------------
face_model = insightface.app.FaceAnalysis(name="buffalo_l")
face_model.prepare(ctx_id=0)

# ----------------------------------------
# FineTuner Network
# ----------------------------------------
class FineTuner(nn.Module):
    def __init__(self, input_dim=512, embedding_dim=128):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        return self.network(x)
    # ----------------------------------------
# Contrastive Loss
# ----------------------------------------
def contrastive_loss(output1, output2, label, margin=1.0):
    """
    Contrastive loss for Siamese Network.

    label = 1 -> Same person
    label = 0 -> Different person
    """

    distance = nn.functional.pairwise_distance(output1, output2)

    loss = torch.mean(
        label * torch.pow(distance, 2) +
        (1 - label) *
        torch.pow(torch.clamp(margin - distance, min=0.0), 2)
    )

    return loss


# ----------------------------------------
# Extract Face Embedding
# ----------------------------------------
def get_embedding(image_path):
    """
    Extract 512-D embedding using InsightFace.
    """

    image = cv2.imread(image_path)

    if image is None:
        return None

    faces = face_model.get(image)

    if len(faces) == 0:
        return None

    embedding = torch.tensor(
        faces[0].embedding,
        dtype=torch.float32
    )

    return embedding
# ----------------------------------------
# Create Random Training Pairs
# ----------------------------------------
def create_pairs(client_path,
                 max_positive_pairs=1000,
                 max_negative_pairs=1000):
    """
    Create balanced random positive and negative pairs.

    Args:
        client_path: Path to client folder
        max_positive_pairs: Number of same-person pairs
        max_negative_pairs: Number of different-person pairs

    Returns:
        pairs, labels
    """

    embeddings = {}

    print("\nExtracting embeddings...")

    # ----------------------------------------
    # Load embeddings for each person
    # ----------------------------------------
    for person in sorted(os.listdir(client_path)):

        person_folder = os.path.join(client_path, person)

        if not os.path.isdir(person_folder):
            continue

        person_embeddings = []

        for image in os.listdir(person_folder):

            image_path = os.path.join(person_folder, image)

            emb = get_embedding(image_path)

            if emb is not None:
                person_embeddings.append(emb)

        embeddings[person] = person_embeddings

        print(f"{person}: {len(person_embeddings)} embeddings")

    pairs = []
    labels = []

    people = list(embeddings.keys())

    # ----------------------------------------
    # Positive Pairs
    # ----------------------------------------
    print("\nGenerating positive pairs...")

    positive = 0

    while positive < max_positive_pairs:

        person = random.choice(people)

        if len(embeddings[person]) < 2:
            continue

        emb1, emb2 = random.sample(embeddings[person], 2)

        pairs.append((emb1, emb2))

        labels.append(1)      # Same person

        positive += 1

    # ----------------------------------------
    # Negative Pairs
    # ----------------------------------------
    print("Generating negative pairs...")

    negative = 0

    while negative < max_negative_pairs:

        person1, person2 = random.sample(people, 2)

        if len(embeddings[person1]) == 0:
            continue

        if len(embeddings[person2]) == 0:
            continue

        emb1 = random.choice(embeddings[person1])

        emb2 = random.choice(embeddings[person2])

        pairs.append((emb1, emb2))

        labels.append(0)      # Different person

        negative += 1

    print(f"\nCreated {len(pairs)} training pairs")

    return pairs, labels
# ----------------------------------------
# Train Local Model
# ----------------------------------------
def train_local_model(
    model,
    client_path,
    epochs=5,
    batch_size=32,
    learning_rate=1e-3,
    positive_pairs=1000,
    negative_pairs=1000
):
    """
    Train the FineTuner model on one client's dataset.

    Args:
        client_path: Path to client folder
        epochs: Number of local training epochs
        batch_size: Batch size
        learning_rate: Optimizer learning rate

    Returns:
        Trained model
    """

    print(f"\n==============================")
    print(f"Training Client: {client_path}")
    print(f"==============================")

    # Create Model
    model = model.to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    # Create Training Pairs
    pairs, labels = create_pairs(
        client_path,
        max_positive_pairs=positive_pairs,
        max_negative_pairs=negative_pairs
    )

    # Convert to tensors
    data1 = torch.stack([pair[0] for pair in pairs])
    data2 = torch.stack([pair[1] for pair in pairs])

    labels = torch.tensor(
        labels,
        dtype=torch.float32
    )

    dataset = TensorDataset(
        data1,
        data2,
        labels
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Training Loop
    model.train()

    for epoch in range(epochs):

        total_loss = 0.0

        progress = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{epochs}"
        )

        for emb1, emb2, label in progress:

            emb1 = emb1.to(DEVICE)
            emb2 = emb2.to(DEVICE)
            label = label.to(DEVICE)

            output1 = model(emb1)
            output2 = model(emb2)

            loss = contrastive_loss(
                output1,
                output2,
                label
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

    print("\nLocal training completed.\n")

    return model
# ----------------------------------------
# Get Model Parameters
# ----------------------------------------
def get_parameters(model):

    return [
        val.cpu().numpy()
        for _, val in model.state_dict().items()
    ]


# ----------------------------------------
# Set Model Parameters
# ----------------------------------------
def set_parameters(model, parameters):

    state_dict = model.state_dict()

    new_state = {}

    for key, value in zip(state_dict.keys(), parameters):

        new_state[key] = torch.tensor(value)

    model.load_state_dict(new_state)

    return model