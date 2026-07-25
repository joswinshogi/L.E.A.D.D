import flwr as fl
from collections import OrderedDict
import torch

from src.trainer import (
    FineTuner,
    train_local_model,
    get_parameters,
    set_parameters,
)

class FaceClient(fl.client.NumPyClient):

    def __init__(self, client_path):
        self.client_path = client_path
        self.model = FineTuner()

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):

        print(f"\nTraining {self.client_path}")

        # Load global model weights
        set_parameters(self.model, parameters)

        # Train locally
        self.model = train_local_model(
            self.model,
            client_path=self.client_path,
            epochs=1,
            batch_size=16,
            positive_pairs=200,
            negative_pairs=200,
        )

        # Return updated weights
        return (
            get_parameters(self.model),
            1,
            {}
        )

    def evaluate(self, parameters, config):

        set_parameters(self.model, parameters)

        loss = 0.0

        accuracy = 0.0

        return loss, 1, {"accuracy": accuracy}


def start_client(client_path):

    client = FaceClient(client_path)

    fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=client.to_client(),
    )