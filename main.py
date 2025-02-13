import argparse
import numpy as np
import flwr as fl
import time
from data_preprocessing import prepare_modified_data
from BiasCorr.BiasCorr_train_test import BiasCorr_test
from FL_MNAR.client import generate_full_client_fn
from FL_MNAR.server import get_full_evaluate_fn, fit_config, FL_MNAR

if __name__ == "__main__":

    # Get command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        required=True
    )
    parser.add_argument(
        '--data_dir',
        type=str
    )
    parser.add_argument(
        '--scheduler',
        type=int,
        required=True
    )
    parser.add_argument(
        '--alpha',
        type=float,
        required=True
    )

    args = parser.parse_args()
    dataset = args.dataset
    data_dir = args.data_dir
    T_s = args.scheduler
    alpha = args.alpha

    T = 100
    E = 5

    if dataset == "Adult":
        C, S, J, K = 10, 2, 20, 34
    elif dataset == "Drug":
        C, S, J, K = 4, 1, 5, 12

    # Initialize local training sets and testing set
    training_sets, val_sets, testing_set = prepare_modified_data(dataset=dataset, data_dir=data_dir, num_partitions=C, batch_size=32, J=J)

    # Initialize client callback based on modified training sets
    full_client_fn_callback = generate_full_client_fn(
        training_sets, K, J, E, T_s, alpha, device='cpu'
    )

    # Initialize prediction parameters
    beta = np.zeros((1, testing_set.shape[1] - 1))

    # Initialize pi_2k parameters for the server
    pi_2k_client = np.concatenate((
        np.ones((1, J)) * np.log(0.9999), np.ones((1, K - J)) * np.log(0.5)
    ), 1)
    pi_2k = pi_2k_client.repeat(C, axis=0)

    # Initialize other server parameters
    pi_2k_c = 0.0001
    cid = -1

    # Declare FL-MNAR strategy
    full_strategy = FL_MNAR(fraction_fit=S / C,  # let's sample 10% of the client each round to do local training
                            min_available_clients=C,  # total number of clients available in the experiment
                            evaluate_fn=get_full_evaluate_fn(testing_set, J),
                            on_fit_config_fn=fit_config,
                            initial_parameters=fl.common.ndarrays_to_parameters([beta, pi_2k_c, pi_2k, cid])
                            )

    # Execute T communication rounds between server and clients
    start = time.perf_counter()
    full_history = fl.simulation.start_simulation(
        client_fn=full_client_fn_callback,  # a callback to construct a client
        num_clients=C,  # total number of clients in the experiment
        config=fl.server.ServerConfig(num_rounds=T),  # let's run for T_s rounds
        strategy=full_strategy,  # the strategy that will orchestrate the whole FL pipeline
    )
    end = time.perf_counter()

    print(f"FL training completed in {end - start:0.4f} seconds")

    # Get testing accuracy and average client training accuracy
    test_acc = [100.0*data[1] for data in full_history.metrics_centralized['accuracy'][1:]][-1]

    val_accs = []
    final_parameters = full_history.metrics_centralized['parameters'][1][-1]

    for val_set in val_sets:
        _, val_acc = BiasCorr_test(beta=final_parameters, testing_set=val_set, J=J)
        val_accs.append(val_acc)

    avg_clt_train_acc = sum(val_accs) / len(val_accs)

    print(f"Average client training accuracy: {avg_clt_train_acc:0.4f}")
    print(f"Testing accuracy: {test_acc:0.4f}")


