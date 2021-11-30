import os
import torch
import numpy as np
import copy
import time
from scipy.optimize import linear_sum_assignment
from train_ddpg import train
from DataPrep import dataPrep
from env_ddpg import envCB
from clustering import kMeans_kNN, KMeans_only
from function_lib import bf_gain_cal, corr_mining
from DDPG_classes import Actor, Critic, OUNoise, init_weights
import pickle


if __name__ == '__main__':

    options = {
        'gpu_idx': 1,
        'num_ant': 32,
        'num_bits': 4,
        'num_NNs': 8,
        'ch_sample_ratio': 0.5,
        'num_loop': 10000,  # outer loop
        'target_update': 3,
        'pf_print': 1000,
        'path': './grid1101-1400.mat',
        'clustering_mode': 'random',
        'save_freq': 50000
    }

    train_opt = {
        'state': 0,
        'best_state': 0,
        'num_iter': 100,
        'tau': 1e-2,
        'overall_iter': 1,
        'replay_memory': [],
        'replay_memory_size': 8192,
        'minibatch_size': 1024,
        'gamma': 0
    }

    ch = dataPrep(options['path'])  # numpy.ndarray: (#users, 128)
    ch = np.concatenate((ch[:, :options['num_ant']],
                         ch[:, int(ch.shape[1] / 2):int(ch.shape[1] / 2) + options['num_ant']]), axis=1)

    u_classifier, sensing_beam = KMeans_only(ch, options['num_NNs'], n_rand_beam=30)
    sensing_beam = torch.from_numpy(sensing_beam).float().cuda()

    filename = 'kmeans_model.sav'
    pickle.dump(u_classifier, open(filename, 'wb'))

    loaded_model = pickle.load(open(filename, 'rb'))

    for sample_id in range(options['num_loop']):

        # ---------- Sampling ---------- #
        n_sample = int(ch.shape[0] * options['ch_sample_ratio'])
        ch_sample_id = np.random.permutation(ch.shape[0])[0:n_sample]
        ch_sample = torch.from_numpy(ch[ch_sample_id, :]).float().cuda()

        # ---------- Clustering ---------- #
        bf_mat_sample = bf_gain_cal(sensing_beam, ch_sample)
        f_matrix = corr_mining(bf_mat_sample)
        f_matrix_np = torch.Tensor.cpu(f_matrix).numpy()
        labels = loaded_model.predict(np.transpose(f_matrix_np))

        user_group = []  # order: clusters
        ch_group = []  # order: clusters
        len_group = []
        for ii in range(options['num_NNs']):
            user_group.append(np.where(labels == ii)[0].tolist())
            ch_group.append(ch_sample[user_group[ii], :])
        print(sample_id)

