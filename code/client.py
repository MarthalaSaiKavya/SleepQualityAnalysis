import warnings
import flwr as fl
import numpy as np
import argparse
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.impute import SimpleImputer

# Import load_sleep_data and other necessary functions from utils
import utils

if __name__ == "__main__":
    N_CLIENTS = 20

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--node-id",
        type=int,
        choices=range(0, N_CLIENTS),
        required=True,
        help="Specifies the artificial data partition",
    )
    args = parser.parse_args()
    partition_id = args.node_id

    (X_train, y_train), (X_test, y_test) = utils.load_sleep_data()

    data_len = len(X_train)
    start_idx = int((partition_id / N_CLIENTS) * data_len)
    end_idx = int(((partition_id + 1) / N_CLIENTS) * data_len)

    X_train = X_train.reshape(-1, X_train.shape[-1])
    X_train = np.array(X_train)

    X_train_partition = np.concatenate([X_train[:start_idx], X_train[end_idx:]])
    X_test_partition = X_train[start_idx:end_idx]
    y_train_partition = np.concatenate([y_train[:start_idx], y_train[end_idx:]]).flatten()
    y_test_partition = y_train[start_idx:end_idx]

    model = LogisticRegression(
        penalty="l2",
        max_iter=1,  # Adjust local epoch as needed
        warm_start=True,
    )

    utils.set_initial_params(model)

    class SleepClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            return utils.get_model_parameters(model)

        def fit(self, parameters, config):
            utils.set_model_params(model, parameters)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train_partition, y_train_partition)
            print(f"Training finished for round {config['server_round']}")
            
            # Get the updated model parameters
            updated_params = utils.get_model_parameters(model)
            
            # Increment the partition_id for the next round
            config['partition_id'] = (config['partition_id'] + 1) % N_CLIENTS
            
            return updated_params, len(X_train_partition), {}

        def evaluate(self, parameters, config):
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test_partition, model.predict_proba(X_test_partition))
            accuracy = model.score(X_test_partition, y_test_partition)
            print({"accuracy": accuracy}, {"loss": loss})
            return loss, len(X_test_partition), {"accuracy": accuracy}

    fl.client.start_numpy_client(server_address="127.0.0.1:8081", client=SleepClient())