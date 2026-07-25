# Federated Face Recognition on Non-IID CCTV Data

Privacy-preserving distributed face recognition using **Federated Learning**, **InsightFace**, **PyTorch**, and **Flower**.

---

## Overview

This project explores **federated face recognition** on distributed CCTV-style face datasets.  
The system uses the pretrained **InsightFace Buffalo_L** model to extract facial embeddings and a lightweight **FineTuner** network to refine those embeddings for face verification.

Training is performed in a federated setting using the **Flower** framework and the **FedAvg** aggregation strategy.  
This allows multiple clients to collaboratively improve a shared model **without sharing raw facial images**.

---

## Key Features

- Federated learning across multiple clients
- Non-IID dataset partitioning
- InsightFace Buffalo_L as the backbone feature extractor
- Lightweight FineTuner network for embedding refinement
- Contrastive loss for face verification training
- Flower-based client-server orchestration
- Evaluation using accuracy, precision, recall, F1-score, ROC-AUC, and similarity distributions
- LaTeX report source and presentation files included

---

## Project Structure

```text
face-recognition-federated/
├── README.md
├── requirements.txt
├── run_client.py
├── src/
│   ├── trainer.py
│   ├── flower_client.py
│   ├── flower_server.py
│   ├── evaluate_global.py
│   ├── split_clients.py
│   ├── create_train_eval.py
│   └── fine_tuning.py
├── report/
│   ├── main.tex
│   ├── references.bib
│   ├── sections/
│   └── figures/
├── presentation/
│   └── Federated_Face_Recognition_Presentation.pptx
└── outputs/
