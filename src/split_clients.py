import os
import shutil
import random

# -----------------------------
# Configuration
# -----------------------------
SOURCE_DIR = "dataset/train"          # Original dataset
OUTPUT_DIR = "clients"

NUM_CLIENTS = 4
TRAIN_RATIO = 0.8
RANDOM_SEED = 42

random.seed(RANDOM_SEED)


def create_client_dirs(person_name):
    """
    Create person folders inside every client.
    """
    for i in range(NUM_CLIENTS):
        path = os.path.join(
            OUTPUT_DIR,
            f"client{i+1}",
            person_name
        )
        os.makedirs(path, exist_ok=True)


def split_images(images):
    """
    Split images equally among clients.
    """

    random.shuffle(images)

    chunk_size = len(images) // NUM_CLIENTS

    splits = []

    for i in range(NUM_CLIENTS):

        start = i * chunk_size

        if i == NUM_CLIENTS - 1:
            end = len(images)
        else:
            end = (i + 1) * chunk_size

        splits.append(images[start:end])

    return splits


def process_dataset():

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    persons = sorted(os.listdir(SOURCE_DIR))

    for person in persons:

        person_path = os.path.join(SOURCE_DIR, person)

        if not os.path.isdir(person_path):
            continue

        print(f"Processing {person}")

        create_client_dirs(person)

        images = [
            img for img in os.listdir(person_path)
            if img.lower().endswith(
                (".jpg", ".jpeg", ".png")
            )
        ]

        client_splits = split_images(images)

        for client_idx, client_images in enumerate(client_splits):

            destination = os.path.join(
                OUTPUT_DIR,
                f"client{client_idx+1}",
                person
            )

            for image in client_images:

                src = os.path.join(person_path, image)
                dst = os.path.join(destination, image)

                shutil.copy2(src, dst)

    print("\nDataset successfully split into clients.")


if __name__ == "__main__":
    process_dataset()