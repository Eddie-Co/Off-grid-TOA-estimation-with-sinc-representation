import torch
import numpy as np
import torch.utils.data as data_utils
from data import re, data
from sklearn.model_selection import train_test_split


def train_test_data(num_samples, signal_dim, n_tau, min_sep,
                    kernel_type, kernel_param, batch_size, xgrid, snr):
    signals, tau = data.gen_signal(num_samples, signal_dim, n_tau, min_sep, snr)
    s_train, s_test, tau_train, tau_test = train_test_split(signals, tau, test_size=0.01, random_state=42)
    train_tau_representation = re.re_tau(tau_train, xgrid, kernel_type, s_train, kernel_param)
    test_tau_representation = re.re_tau(tau_test, xgrid, kernel_type, s_train, kernel_param)
    s_train=torch.from_numpy(s_train).float()
    s_test=torch.from_numpy(s_test).float()
    tau_test=torch.from_numpy(tau_test).float()
    train_tau_representation=torch.from_numpy(train_tau_representation).float()
    test_tau_representation = torch.from_numpy(test_tau_representation).float()
    train_dataset = data_utils.TensorDataset(s_train, train_tau_representation)
    test_dataset = data_utils.TensorDataset(s_test, test_tau_representation, tau_test)
    train_data = data_utils.DataLoader(train_dataset, batch_size=batch_size)
    test_data = data_utils.DataLoader(test_dataset, batch_size=batch_size)
    return train_data, test_data


def make_train_test_data(args):
    xgrid = np.linspace(0, 20, args.tau_size, endpoint=False)
    kernel_param = args.gaussian_std / args.signal_dim
    return train_test_data(args.nums_data, signal_dim=args.signal_dim, n_tau=args.n_tau,
                           min_sep=args.min_sep, kernel_type=args.kernel_type, kernel_param=kernel_param,
                           batch_size=args.batch_size, xgrid=xgrid, snr=args.snr)

