"""
Training and testing for Seq2Seq-attn-R.
Author: Valentina Rizzello
email: valentina.rizzello@tum.de

"""
import os
import random

import numpy as np

from models.seq2seq_attn_r import Seq2SeqAttnR
from training.common_func import *


def seq2seq_decoding(model, data_loader, device):
    model.eval()

    pred_channels = []

    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            data = data.to(device)
            target = target[:, 1:, :].to(device)

            max_len = target.shape[1]
            output = model.predict(data, max_len)

            pred_channels.append(output)

    return torch.cat(pred_channels, dim=0)


def train_epoch(model, optimizer, data_loader, device):
    model.train()

    for i, (data, target) in enumerate(data_loader):
        data = data.to(device)
        dec_input = target[:, :-1:, :].to(device)
        target = target[:, 1:, :].to(device)

        optimizer.zero_grad()

        output = model(data, dec_input)

        loss, _, _ = nmse_eval(target, output)
        loss.backward()
        optimizer.step()


def seq2seq_attn_r_main(config, config_paths, device, training=False):
    # for reproducibility
    random.seed(1234)
    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.use_deterministic_algorithms(True)

    foldername = config_paths["results"] + 'Seq2Seq_attn_R_snr_' + f'{config["snr"]}'
    if not os.path.exists(foldername):
        os.mkdir(foldername)

    snr = config["snr"]
    # load data
    x, y, target = load_data(config_paths["csi"], noise=True, snr=snr, model="seq2seq")
    x_train, x_val, x_test = x
    y_train, y_val, y_test = y
    _, _, y_test_c = target

    # transform to tensor
    x_train = transform_to_tensor(x_train)
    x_val = transform_to_tensor(x_val)

    y_train = transform_to_tensor(y_train)
    y_val = transform_to_tensor(y_val)

    # create dataloader
    train_data_set = TensorDataset(x_train, y_train)
    val_data_set = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_data_set, batch_size=config["batch_size"], shuffle=True)
    validation_loader = DataLoader(val_data_set, batch_size=200, shuffle=False)

    model = Seq2SeqAttnR(
        input_size=64,
        hidden_size=config["hidden_dim"],
        output_size=64,
        n_layers=config["n_layers"],
        max_length=16
    ).to(device)

    if training:
        # training
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        n_epochs = config["epochs"]

        for epoch in range(1, n_epochs + 1):
            print('Epoch:', epoch)
            train_epoch(model, optimizer, train_loader, device)
            # TODO: save optimal model based on validation set

    # testing

    # load best model
    model.load_state_dict(torch.load(os.path.join(foldername, 'checkpoint.pt')))
    model.eval()

    print("Testing l=16 and delta=4 (same as training): ", end="")
    predict_and_evaluate(x_test, y_test, y_test_c, model, device, snr, seq2seq_decoding)

    print("Testing l=8 and delta=2: ", end="")
    test_with_lengths(8, 2, config_paths["csi"], snr, model, device,
                      model_name="seq2seq", output_fun=seq2seq_decoding)

    print("Testing l=14 and delta=6: ", end="")
    test_with_lengths(14, 6, config_paths["csi"], snr, model, device,
                      model_name="seq2seq", output_fun=seq2seq_decoding)

    print("---------------------------------------------------------------------------")
