"""
Training and testing for Transformer-RPE.
Author: Valentina Rizzello
email: valentina.rizzello@tum.de

"""
import os
import random

import numpy as np

from models.transformer_rpe import *
from training.common_func import *


def dynamic_decoding(model, data_loader, device, src_mask=None):
    model.eval()

    pred_channels = []

    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            data = data.to(device)
            target = target[:, 1:, :].to(device)

            e = model.enc(data, src_mask)

            init_sample = data[:, -1, :].unsqueeze(1)
            input_seq = init_sample

            max_len = 20
            pred_len = target.size(1)

            for j in range(pred_len):
                curr_len = input_seq.size(1)
                mask = buffered_future_mask(curr_len, max_len, device)
                output = model.dec(input_seq, e, mask)

                input_seq = torch.cat([init_sample, output], dim=1)

            pred_channels.append(output)

    return torch.cat(pred_channels, dim=0)


def train_epoch(model, optimizer, data_loader, mask, device, src_mask=None):
    model.train()

    for i, (data, target) in enumerate(data_loader):
        data = data.to(device)
        dec_input = target[:, :-1:, :].to(device)
        target = target[:, 1:, :].to(device)

        optimizer.zero_grad()

        output = model(data, dec_input, mask, src_mask)

        loss, _, _ = nmse_eval(target, output)
        loss.backward()
        optimizer.step()


def transformer_rpe_main(config, config_paths, device, training=False):
    # for reproducibility
    random.seed(1234)
    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.use_deterministic_algorithms(True)

    foldername = config_paths["results"] + 'Transformer_RPE_snr_' + f'{config["snr"]}'
    if not os.path.exists(foldername):
        os.mkdir(foldername)

    snr = config["snr"]
    # load data
    x, y, target = load_data(config_paths["csi"], noise=True, snr=snr, model="tnn")
    x_train, x_val, x_test = x
    y_train, y_val, y_test = y
    _, y_val_c, y_test_c = target  # target data clean without any additive noise

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

    max_len = 20
    len_out = y_train.size(1) - 1
    trg_mask = buffered_future_mask(curr_len=len_out, max_len=max_len, device=device)

    model = TransformerRPE(
        obs_dim=64,
        heads=config["h"],
        hidden_dim=config["hidden_dim"],
        n_layers_enc=config["n_layers_enc"],
        n_layers_dec=config["n_layers_dec"],
        p_enc=config["p_enc"],
        w=config["w"]
    ).to(device)

    if training:
        # training
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        n_epochs = config["epochs"]

        for epoch in range(1, n_epochs + 1):
            print('Epoch:', epoch)
            train_epoch(model, optimizer, train_loader, trg_mask, device)
            # TODO: save optimal model based on validation set

    # testing

    # load best model
    model.load_state_dict(torch.load(os.path.join(foldername, 'checkpoint.pt')))
    model.eval()

    print("Testing l=16 and delta=4 (same as training): ", end="")
    predict_and_evaluate(x_test, y_test, y_test_c, model, device, snr, dynamic_decoding)

    print("Testing l=8 and delta=2: ", end="")
    test_with_lengths(8, 2, config_paths["csi"], snr, model, device,
                      model_name="tnn", output_fun=dynamic_decoding)

    print("Testing l=14 and delta=6: ", end="")
    test_with_lengths(14, 6, config_paths["csi"], snr, model, device,
                      model_name="tnn", output_fun=dynamic_decoding)

    print("---------------------------------------------------------------------------")
