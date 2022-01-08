import numpy as np
import operator as op
from functools import reduce
import torch


def phases_gen(q_bits):
    angles = np.linspace(0, 2 * np.pi, 2 ** q_bits, endpoint=False)
    cb = np.exp(1j * angles)
    codebook = np.zeros((2 ** q_bits, 2))  # shape of the codebook
    for idx in range(cb.shape[0]):
        codebook[idx, 0] = np.real(cb[idx])
        codebook[idx, 1] = np.imag(cb[idx])
    return codebook


def bf_gain_cal_np(cb, H):

    n_ant = int(H.shape[1] / 2)

    bf_r = cb[:, ::2]
    bf_i = cb[:, 1::2]
    ch_r = H[:, :n_ant]
    ch_i = H[:, n_ant:]

    bf_gain_1 = np.matmul(bf_r, ch_r.transpose())
    bf_gain_2 = np.matmul(bf_i, ch_i.transpose())
    bf_gain_3 = np.matmul(bf_r, ch_i.transpose())
    bf_gain_4 = np.matmul(bf_i, ch_r.transpose())
    bf_gain_r = (bf_gain_1 + bf_gain_2) ** 2
    bf_gain_i = (bf_gain_3 - bf_gain_4) ** 2

    bf_gain_pattern = bf_gain_r + bf_gain_i  # BF gain pattern matrix

    return bf_gain_pattern


def bf_gain_cal(cb, H):
    
    n_ant = int(H.size(1) / 2)
    
    bf_r = cb[:, ::2]
    bf_i = cb[:, 1::2]
    ch_r = H[:, :n_ant]
    ch_i = H[:, n_ant:]
    
    bf_gain_1 = torch.matmul(bf_r, torch.t(ch_r))
    bf_gain_2 = torch.matmul(bf_i, torch.t(ch_i))
    bf_gain_3 = torch.matmul(bf_r, torch.t(ch_i))
    bf_gain_4 = torch.matmul(bf_i, torch.t(ch_r))
    bf_gain_r = (bf_gain_1 + bf_gain_2) ** 2
    bf_gain_i = (bf_gain_3 - bf_gain_4) ** 2
    
    bf_gain_pattern = bf_gain_r + bf_gain_i  # BF gain pattern matrix
    
    return bf_gain_pattern


def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return int(numer / denom)


def corr_mining_np(bf_gain_matrix):
    # norm_factor = np.sqrt(np.mean(bf_gain_matrix, axis=0))
    # feature_mat = np.zeros(bf_gain_matrix.shape)
    # for u_id in range(bf_gain_matrix.shape[1]):
    #     feature_mat[:, u_id] = bf_gain_matrix[:, u_id] / norm_factor[u_id]
    num_raw_feature = bf_gain_matrix.shape[0]
    num_user = bf_gain_matrix.shape[1]
    num_feature = ncr(num_raw_feature, 2)
    norm_factor = np.mean(bf_gain_matrix, axis=0)
    feature_mat = np.zeros((num_feature, num_user))
    for u_id in range(num_user):
        feature_count = 0
        for idx_1 in range(num_raw_feature - 1):
            for idx_2 in range(idx_1 + 1, num_raw_feature):
                feature_mat[feature_count, u_id] = (bf_gain_matrix[idx_1, u_id] - bf_gain_matrix[idx_2, u_id]) / \
                                                   norm_factor[u_id]
                feature_count = feature_count + 1
        if feature_count != num_feature:
            print('error...')
    return feature_mat


def corr_mining(bf_gain_matrix):
    num_sen_beams = bf_gain_matrix.shape[0]
    num_user = bf_gain_matrix.shape[1]
    num_feature = ncr(num_sen_beams, 2)
    norm_factor = torch.mean(bf_gain_matrix, dim=0)
    mat_1 = torch.zeros((num_feature, num_user)).float().cuda()
    mat_2 = torch.zeros((num_feature, num_user)).float().cuda()
    for ii in range(num_sen_beams-1):
        if ii == 0:
            mat_1 = bf_gain_matrix[ii, :].reshape(1, -1).repeat(num_sen_beams-ii-1, 1)
            mat_2 = bf_gain_matrix[ii+1:, :]
        else:
            mat_1 = torch.cat((mat_1, bf_gain_matrix[ii, :].reshape(1, -1).repeat(num_sen_beams-ii-1, 1)))
            mat_2 = torch.cat((mat_2, bf_gain_matrix[ii+1:, :]))
    feature_mat = torch.div(mat_1 - mat_2, norm_factor)
    return feature_mat


def proj_pattern_cal(best_beam, sensing_beam):
    best_beam_r = best_beam[:, ::2]
    best_beam_i = best_beam[:, 1::2]
    sensing_beam_r = sensing_beam[:, ::2]
    sensing_beam_i = sensing_beam[:, 1::2]
    bf_gain_1 = np.matmul(best_beam_r, np.transpose(sensing_beam_r))
    bf_gain_2 = np.matmul(best_beam_i, np.transpose(sensing_beam_i))
    bf_gain_3 = np.matmul(best_beam_r, np.transpose(sensing_beam_i))
    bf_gain_4 = np.matmul(best_beam_i, np.transpose(sensing_beam_r))
    bf_gain_r = (bf_gain_1 + bf_gain_2) ** 2
    bf_gain_i = (bf_gain_3 - bf_gain_4) ** 2
    proj_pattern = bf_gain_r + bf_gain_i
    return proj_pattern
