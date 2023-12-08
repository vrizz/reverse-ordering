from torch.utils.data import TensorDataset, DataLoader

from utils.dataset_utils import *
from utils.loader import load_data
from utils.metrics import *


def predict_and_evaluate(x_test, y_test, y_test_c, model, device, snr, get_output):
    x_test = transform_to_tensor(x_test)
    y_test = transform_to_tensor(y_test)
    y_test_c = transform_to_tensor(y_test_c)

    test_data_set = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_data_set, batch_size=200, shuffle=False)

    pred_channels = get_output(model, test_loader, device)

    # we conpute the nmse with respect to the clean target of the test set
    nmse_avg, mse_avg, nmse_all = nmse_eval(y_test_c.to(device), pred_channels)

    print('SNR=' + str(snr) + 'dB', 'NMSE=' + str(round(nmse_avg.item(), 4)), sep="   ")


def test_with_lengths(n_inputs, n_outputs, csi_path, snr, model, device, model_name, output_fun):
    x, y, target = load_data(csi_path, noise=True, snr=snr, model=model_name,
                             n_inputs=n_inputs, n_outputs=n_outputs)

    x_train, x_val, x_test = x
    y_train, y_val, y_test = y
    _, _, y_test_c = target

    predict_and_evaluate(x_test, y_test, y_test_c, model, device, snr, output_fun)
