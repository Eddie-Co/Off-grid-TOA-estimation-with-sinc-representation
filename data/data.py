import numpy as np
from tqdm import tqdm


def tau_generator(tau, ntau, min_sep):
    random_tau(tau, ntau, min_sep)


def random_tau(tau, ntau, min_sep):
    """
    Generate time delay values uniformly.
    """
    for i in range(ntau):
        tau_new = np.random.rand()
        condition = True
        while condition:
            tau_new = np.random.rand()
            condition = (np.min(np.abs(tau - tau_new)) < min_sep)
        tau[i] = tau_new


def noisy(s, snr):
    snr = 10 ** (snr / 10)
    a = s.shape[0]
    nr = np.zeros((a))
    xpower = np.sum(s ** 2) / a
    npower = xpower / snr
    nr = np.random.randn(a) * np.sqrt(npower)
    return nr


def gen_signal(num_samples, signal_dim, num_tau, min_sep, snr):
    np.random.seed(2)
    f0 = 300e4
    k1 = 3e9
    s = np.zeros((num_samples, 2, signal_dim))
    # 采样函数
    tt = np.linspace(-0.0025, 0.0025, 50)
    tau = np.ones((num_samples, num_tau)) * np.inf
    sin = np.ones(signal_dim, dtype=complex)
    d_sep = min_sep / signal_dim
    nfreq = num_tau
    for n in tqdm(range(num_samples)):
        tau_generator(tau[n], nfreq, d_sep)
        for i in range(nfreq):
            for k in range(signal_dim):
                sin[k] = np.exp(2j * np.pi * f0 * (tt[k] - tau[n, i] * 20 * 10e-8) + 1j * np.pi * k1 *
                                (tt[k] - tau[n, i] * 20 * 10e-8) ** 2)
            s[n, 0] = s[n, 0] + sin.real
            s[n, 1] = s[n, 1] + sin.imag
        s[n, 0] = s[n, 0] + noisy(s[n, 0], snr)  #noise改变!!!!!
        s[n, 1] = s[n, 1] + noisy(s[n, 1], snr)
        s[n] = s[n] / np.sqrt(np.mean(np.power(s[n], 2)))
    tau.sort(axis=1)
    tau = tau * 20
    return s.astype('float32'), tau.astype('float32')

