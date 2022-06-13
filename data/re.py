import numpy as np
import scipy.signal
import time

def re_tau(f, xgrid, kernel_type, signal, param=None):
    """
    Coding the TOA values to the representation.
    """
    if kernel_type == 'gaussian':
        return gaussian_kernel(f, xgrid, param)
    elif kernel_type == 'onehot':
        return onehot(f, xgrid, param)
    elif kernel_type == 'sinc':
        return sinc(f, xgrid, param)
    elif kernel_type == 'corr':
        return corr(f, xgrid, signal)


def onehot(f, xgrid, sigma):
    """
    Encoding TOA values with one-hot.
    """
    fr = np.zeros((f.shape[0], xgrid.shape[0]))
    for i in range(f.shape[1]):
        dist = np.abs(xgrid[None, :] - f[:, i][:, None])
        fr += np.where(dist < 0.01, 1, 0)
    return fr


def corr(f, xgrid, signal):
    """
    Encoding TOA values with correlation.
    """
    start_time = time.time()
    fr = np.zeros((f.shape[0], xgrid.shape[0]))
    tt = np.linspace(-0.0025, 0.0025, 50)

    for j in range(f.shape[0]):
        for i in range(xgrid.shape[0]):
            fr[j, i] = np.sum(signal[j, 0]*np.exp(2j * np.pi * 300e4 * (tt - xgrid[i] * 10e-8) + 1j * np.pi * 3e9 * (tt - xgrid[i] * 10e-8) ** 2).real+
                    signal[j, 1]*np.exp(2j * np.pi * 300e4 * (tt - xgrid[i] * 10e-8) + 1j * np.pi * 3e9 * (tt - xgrid[i] * 10e-8) ** 2).imag)
            if fr[j, i]<20:
                fr[j, i] = 0
    end_time = time.time()-start_time
    print(end_time)
    return fr


def gaussian_kernel(f, xgrid, sigma):
    """
    Encoding TOA values with gaussian kernel.
    """
    start_time = time.time()
    fr = np.zeros((f.shape[0], xgrid.shape[0]))
    fr1 = np.zeros((f.shape[0], xgrid.shape[0]))
    ok = np.zeros((f.shape[0]))
    ok1 = np.zeros((f.shape[0],f.shape[1]))
    for i in range(f.shape[1]):
        dist = xgrid[None, :] - f[:, i][:, None]
        dist1 = np.abs(dist)
        for k in range(f.shape[0]):
            m = np.argmin(np.abs(dist[k]))
            fr[k] = (2-100*dist[k, m]) * np.exp(- dist1[k] ** 2 / sigma ** 2)
        fr1 += fr
    end_time = time.time() - start_time
    print(end_time)
    return fr1


def sinc(f, xgrid, sigma):
    """
    Encoding TOA values with sinc kernel.
    """
    start_time = time.time()
    fr = np.zeros((f.shape[0], xgrid.shape[0]))
    fr1 = np.zeros((f.shape[0], xgrid.shape[0]))
    test1 = np.zeros((f.shape[0], f.shape[1]))
    for i in range(f.shape[1]):
        dist = xgrid[None, :] - f[:, i][:, None]
        dist1 = (dist**2)/100
        for k in range(f.shape[0]):

            test1[k, i] = np.min(np.abs(dist[k]))
            m = np.argmin(np.abs(dist[k]))
            fr[k] = (2-100*dist[k, m]) * np.abs(np.sin(10000*np.pi * dist1[k] + 1e-8)) / (10000 * np.pi * dist1[k] + 1e-8)
        fr1 += fr
    end_time = time.time() - start_time
    print("Coding time : %.4f"%(end_time))
    return fr1


def find_tau(fr, ntau, xgrid, max_tau=3):
    """
    Extract TOA values from encoding TOA values representation by locating the highest peaks.
    """
    tau_1 = -np.ones((ntau.shape[0], max_tau))
    tau_2 = -np.ones((ntau.shape[0], max_tau))
    for n in range(len(ntau)):
        nf = max_tau
        find_peaks_out = scipy.signal.find_peaks(fr[n], height=(None, None), distance=30)
        num_spikes = min(len(find_peaks_out[0]), int(nf))
        idx = np.sort(np.argpartition(find_peaks_out[1]['peak_heights'], -num_spikes)[-num_spikes:])
        tau_1[n, :num_spikes] = xgrid[find_peaks_out[0][idx]]             #TOA values
        tau_2[n, :num_spikes] = find_peaks_out[1]['peak_heights'][idx]   #Calculating the quantization error.
    return tau_1, tau_2


