import os
import random
import shutil

SOURCE_DIR = "Images"
OUTPUT_DIR = "dataset"

TRAIN_RATIO = 0.8

random.seed(42)

os.makedirs(os.path.join(OUTPUT_DIR, "train"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "eval"), exist_ok=True)

for person in os.listdir(SOURCE_DIR):

    person_path = os.path.join(SOURCE_DIR, person)

    if not os.path.isdir(person_path):
        continue

    images = [
        img for img in os.listdir(person_path)
        if img.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    random.shuffle(images)

    split = int(len(images) * TRAIN_RATIO)

    train_images = images[:split]
    eval_images = images[split:]

    train_folder = os.path.join(OUTPUT_DIR, "train", person)
    eval_folder = os.path.join(OUTPUT_DIR, "eval", person)

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(eval_folder, exist_ok=True)

    for img in train_images:
        shutil.copy2(
            os.path.join(person_path, img),
            os.path.join(train_folder, img)
        )

    for img in eval_images:
        shutil.copy2(
            os.path.join(person_path, img),
            os.path.join(eval_folder, img)
        )

print("Dataset split completed.")