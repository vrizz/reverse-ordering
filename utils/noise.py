import numpy as np
from einops import rearrange


def add_noise(h, sigma):
    n = np.random.randn(h.shape[0], h.shape[1], h.shape[2]).astype('float32') * sigma
    y = h + n

    # x = rearrange(h, 'n s a -> (n s) a')
    # E_x = np.mean(np.linalg.norm(x, axis=1) ** 2)
    # n = rearrange(n, 'n s a -> (n s) a')
    # E_n = np.mean(np.linalg.norm(n, axis=1) ** 2)
    # print(E_x / E_n)
    return y


def get_sigma(h, snr=10):
    """

    Args:
        h (n_samples, shapshots, 2*ant): clean data
        snr (int): snr in dB per snapshot

    Returns:

    """

    snr_lin = 10 ** (snr / 10)

    x = rearrange(h, 'n s a -> (n s) a')
    avg_norm_squared = np.mean(np.linalg.norm(x, axis=1) ** 2)
    scale_factor = x.shape[-1]

    sigma = np.sqrt(avg_norm_squared / (snr_lin * scale_factor)).astype('float32')

    return sigma
