import os
import flwr as fl
import torch
import numpy as np

from trainer import FineTuner


class SaveModelStrategy(fl.server.strategy.FedAvg):

    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round,
            results,
            failures,
        )

        if aggregated_parameters is not None:

            print(f"\nSaving global model after round {server_round}...")

            # Convert Flower Parameters -> NumPy arrays
            ndarrays = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            model = FineTuner()

            state_dict = model.state_dict()

            for key, value in zip(state_dict.keys(), ndarrays):
                state_dict[key] = torch.tensor(value)

            model.load_state_dict(state_dict)

            os.makedirs("outputs", exist_ok=True)

            torch.save(
                model.state_dict(),
                "outputs/global_model.pt",
            )

            print("Global model saved!")

        return aggregated_parameters, aggregated_metrics


def start_server():

    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=4,
        min_available_clients=4,
    )

    fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=2),
        strategy=strategy,
    )


if __name__ == "__main__":
    start_server()