import torch
from einops import rearrange


def transform_to_complex(x):
    """
    Input:
        x: input or output of the neural network. shape [-1, 2, n_ant, n_carr]
    Outputs:
        y: complex matrix. shape [-1, n_ant, n_carr]
    """

    x_real = x[:, :, 0:32]
    x_imag = x[:, :, 32:64]

    y = x_real + 1j * x_imag

    y = rearrange(y, 'n s a -> n a s')

    return y


def nmse_eval(y_true, y_pred):
    y_true = y_true.reshape(len(y_true), -1)
    y_pred = y_pred.reshape(len(y_pred), -1)

    mse = torch.sum(abs(y_pred - y_true) ** 2, dim=1)
    power = torch.sum(abs(y_true) ** 2, dim=1)

    nmse = mse / power
    return torch.mean(nmse), torch.mean(mse), nmse


def cos_sim_eval(y_true, y_pred):
    # construct complex number
    y_true_c = transform_to_complex(y_true)
    y_pred_c = transform_to_complex(y_pred)

    # evaluate cos-sim
    den_1 = (abs(y_true_c.conj() * y_true_c)).sum(dim=1).sqrt()
    den_2 = (abs(y_pred_c.conj() * y_pred_c)).sum(dim=1).sqrt()

    num = abs((y_true_c.conj() * y_pred_c).sum(dim=1))

    rho = (num / (den_1 * den_2)).mean(dim=1)

    return rho.mean(), rho


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
