"""
Main script for training and evaluating the models.
"""
import argparse

from training.seq2seq import seq2seq_attn_r_main
from training.tnn import transformer_rpe_main

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Run a specific model")
    parser.add_argument("model", type=str, help="Name of the model to run")
    args = parser.parse_args()

    model_name = args.model

    config_paths = {
        'csi': './dataset/channels.hdf5',
        'results': './checkpoints/',
    }

    snr_list = [-5, 0, 5, 10, 15, 20]
    device = "cuda:1"
    epochs = 500

    config_tnn = {
        'p_enc': 0.0,
        'w': 1,
        'h': 4,
        'hidden_dim': 128,
        'lr': 0.001,
        'batch_size': 200,
        'n_layers_enc': 2,
        'n_layers_dec': 2,
        'epochs': epochs
    }

    config_seq2seq = {
        'hidden_dim': 128,
        'lr': 0.001,
        'batch_size': 200,
        'n_layers': 2,
        'epochs': epochs
    }

    for snr in snr_list:
        config_tnn['snr'] = snr
        config_seq2seq['snr'] = snr

        if model_name == "transformer-rpe":
            transformer_rpe_main(config=config_tnn, config_paths=config_paths, device=device)
        elif model_name == "seq2seq-attn-r":
            seq2seq_attn_r_main(config=config_seq2seq, config_paths=config_paths, device=device)
        else:
            print(f"Model '{model_name}' not recognized.")
