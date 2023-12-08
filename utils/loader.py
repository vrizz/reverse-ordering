import h5py

from utils.noise import *


def _reshape_csi(x):
    """
    Reshape x as [N_samples, N_snapshots, N_ant]
    """
    return rearrange(x, 's a u d -> (d u) s a')


def _load_csi(csi_path, mode="real"):
    f = h5py.File(csi_path, 'r')
    ch_re = np.array(f['H_r']).astype('float32')
    ch_imag = np.array(f['H_i']).astype('float32')

    ch_re = _reshape_csi(ch_re)
    ch_imag = _reshape_csi(ch_imag)

    if mode == "complex":
        X = ch_re + 1j * ch_imag
    else:
        X = np.concatenate((ch_re, ch_imag), axis=-1)

    idx = np.arange(0, len(X))
    np.random.shuffle(idx)

    return X[idx]


def _split_data(x):
    len_data = len(x)

    end_train = int(len_data * 0.8)
    end_val = int(len_data * 0.9)

    x_train = x[0:end_train]
    x_val = x[end_train:end_val]
    x_test = x[end_val:len_data]

    return x_train, x_val, x_test


def _preprocess(H, noise=True, snr=10, model="tnn", n_inputs=16, n_outputs=4, n_history=8):
    if n_inputs + n_inputs != 20:
        H = H[:, :(n_inputs + n_outputs), :]

    target = H[:, -n_outputs:, :]

    if noise is True:
        sigma = get_sigma(H, snr=snr)
        H = add_noise(H, sigma)

    if model == "rnn":
        X = H[:, 0, :-1:, :]
        Y = H[:, -n_outputs:, :]

    else:
        X = H[:, :n_inputs, :]

        if model == "tnn" or model == "seq2seq":
            Y = H[:, -(n_outputs + 1):, :]

        if model == "mlp" or model == "keeplast" or model == "mar" or model == "lstm":
            Y = H[:, -n_outputs:, :]

        if model == "informer":
            Y = H[:, -(n_outputs + n_history):, :]

    return X, Y, target


def load_data(csi_path, noise=True, snr=10, model="tnn", n_inputs=16, n_outputs=4, n_history=8):
    H = _load_csi(csi_path)
    X, Y, target = _preprocess(H=H, noise=noise, snr=snr, model=model, n_inputs=n_inputs, n_outputs=n_outputs,
                              n_history=n_history)

    return _split_data(X), _split_data(Y), _split_data(target)
