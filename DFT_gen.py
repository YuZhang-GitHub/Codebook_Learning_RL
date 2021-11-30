import scipy.io as scio
import numpy as np


def DFT_gen(dft_mode):
    if dft_mode == 'dft':
        dft_mtx = scio.loadmat('./DFT_matrices/dft_16.mat')['W_UPA']  # columns are DFT bf vectors
    elif dft_mode == 'dft_corrupted':
        dft_mtx = scio.loadmat('./DFT_matrices/dft_16_corrupted.mat')['W_UPA_corrupted']  # columns are DFT bf vectors
    else:
        dft_mtx = 0
        ValueError('please check DFT mode.')
    num_beam = dft_mtx.shape[1]
    num_ant = dft_mtx.shape[0]
    dft_mtx_r = np.real(dft_mtx)
    dft_mtx_i = np.imag(dft_mtx)
    dft_cat = np.zeros((num_beam, 2 * num_ant))
    for ii in range(num_beam):
        for jj in range(num_ant):
            dft_cat[ii, 2*jj] = dft_mtx_r[jj, ii]
            dft_cat[ii, 2*jj+1] = dft_mtx_i[jj, ii]

    return dft_cat

# dft_mtx = load_file['W_UPA']
# a = DFT_gen()
# pp = 1
