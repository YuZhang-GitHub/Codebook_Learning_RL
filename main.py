import os
import time
import torch
import numpy as np
import copy
import pickle
from scipy.optimize import linear_sum_assignment
from train_ddpg import train
from DataPrep import dataPrep
from env_ddpg import envCB
from clustering import KMeans_only
from function_lib import bf_gain_cal, corr_mining
from DDPG_classes import Actor, Critic, OUNoise, init_weights


if __name__ == '__main__':

    options = {
        'gpu_idx': 0,
        'num_ant': 32,
        'num_bits': 4,
        'num_NNs': 4,  # codebook size
        'ch_sample_ratio': 0.5,
        'num_loop': 10000,  # outer loop
        'target_update': 3,
        'path': './grid1101-1400.mat',
        'clustering_mode': 'random',
    }

    train_opt = {
        'state': 0,
        'best_state': 0,
        'num_iter': 100,  # inner loop
        'tau': 1e-2,
        'overall_iter': 1,
        'replay_memory': [],
        'replay_memory_size': 8192,
        'minibatch_size': 1024,
        'gamma': 0
    }

    if not os.path.exists('beams/'):
        os.mkdir('beams/')

    ch = dataPrep(options['path'])
    ch = np.concatenate((ch[:, :options['num_ant']],
                         ch[:, int(ch.shape[1] / 2):int(ch.shape[1] / 2) + options['num_ant']]), axis=1)

    with torch.cuda.device(options['gpu_idx']):

        u_classifier, sensing_beam = KMeans_only(ch, options['num_NNs'], n_bit=options['num_bits'], n_rand_beam=30)
        np.save('sensing_beam.npy', sensing_beam)
        sensing_beam = torch.from_numpy(sensing_beam).float().cuda()

        filename = 'kmeans_model.sav'
        pickle.dump(u_classifier, open(filename, 'wb'))

        # Quantization settings
        options['num_ph'] = 2 ** options['num_bits']
        options['multi_step'] = torch.from_numpy(
            np.linspace(int(-(options['num_ph'] - 2) / 2),
                        int(options['num_ph'] / 2),
                        num=options['num_ph'],
                        endpoint=True)).type(dtype=torch.float32).reshape(1, -1).cuda()
        options['pi'] = torch.tensor(np.pi).cuda()
        options['ph_table'] = (2 * options['pi']) / options['num_ph'] * options['multi_step']
        options['ph_table'].cuda()
        options['ph_table_rep'] = options['ph_table'].repeat(options['num_ant'], 1)

        # initialize DRL models
        actor_net_list = []
        critic_net_list = []
        actor_net_t_list = []
        critic_net_t_list = []
        ounoise_list = []
        env_list = []
        train_opt_list = []

        for beam_id in range(options['num_NNs']):
            actor_net_list.append(Actor(options['num_ant'], options['num_ant']))
            actor_net_t_list.append(Actor(options['num_ant'], options['num_ant']))
            critic_net_list.append(Critic(2 * options['num_ant'], 1))
            critic_net_t_list.append(Critic(2 * options['num_ant'], 1))
            ounoise_list.append(OUNoise((1, options['num_ant'])))
            env_list.append(envCB(ch, options['num_ant'], options['num_bits'], beam_id, options))
            train_opt_list.append(copy.deepcopy(train_opt))

            actor_net_list[beam_id] = actor_net_list[beam_id].cuda()
            actor_net_t_list[beam_id] = actor_net_t_list[beam_id].cuda()
            critic_net_list[beam_id] = critic_net_list[beam_id].cuda()
            critic_net_t_list[beam_id] = critic_net_t_list[beam_id].cuda()
            actor_net_list[beam_id].apply(init_weights)
            actor_net_t_list[beam_id].load_state_dict(actor_net_list[beam_id].state_dict())
            critic_net_list[beam_id].apply(init_weights)
            critic_net_t_list[beam_id].load_state_dict(critic_net_list[beam_id].state_dict())

        # start_time = time.time()

        # outer loop for randomly sampling users, emulating user dynamics
        for sample_id in range(options['num_loop']):

            # ---------- Sampling ---------- #
            n_sample = int(ch.shape[0] * options['ch_sample_ratio'])
            ch_sample_id = np.random.permutation(ch.shape[0])[0:n_sample]
            ch_sample = torch.from_numpy(ch[ch_sample_id, :]).float().cuda()

            # ---------- Clustering ---------- #
            start_time = time.time()

            bf_mat_sample = bf_gain_cal(sensing_beam, ch_sample)
            # print("Clustering -1 uses %s seconds." % (time.time() - start_time))
            # start_time = time.time()
            f_matrix = corr_mining(bf_mat_sample)
            f_matrix_np = torch.Tensor.cpu(f_matrix).numpy()
            # print("Clustering 0 uses %s seconds." % (time.time() - start_time))
            # start_time = time.time()
            labels = u_classifier.predict(np.transpose(f_matrix_np).astype(float))

            # print("Clustering 1 uses %s seconds." % (time.time() - start_time))
            # start_time = time.time()

            user_group = []  # order: clusters
            ch_group = []  # order: clusters
            for ii in range(options['num_NNs']):
                user_group.append(np.where(labels == ii)[0].tolist())
                ch_group.append(ch_sample[user_group[ii], :])

            print("Clustering 2 uses %s seconds." % (time.time() - start_time))

            # ---------- Assignment ---------- #
            start_time = time.time()

            # best_state matrix
            best_beam_mtx = torch.zeros((options['num_NNs'], 2 * options['num_ant'])).float().cuda()
            for pp in range(options['num_NNs']):
                best_beam_mtx[pp, :] = env_list[pp].best_bf_vec
            gain_mtx = bf_gain_cal(best_beam_mtx, ch_sample)  # (n_beam, n_user)
            for ii in range(options['num_NNs']):
                if ii == 0:
                    cost_mtx = torch.mean(gain_mtx[:, user_group[ii]], dim=1).reshape(options['num_NNs'], -1)
                else:
                    sub = torch.mean(gain_mtx[:, user_group[ii]], dim=1).reshape(options['num_NNs'], -1)
                    cost_mtx = torch.cat((cost_mtx, sub), dim=1)
            cost_mtx = -torch.Tensor.cpu(cost_mtx).numpy()
            row_ind, col_ind = linear_sum_assignment(cost_mtx)
            assignment_record = dict(zip(row_ind.tolist(), col_ind.tolist()))  # key: network, value: cluster
            print(assignment_record)
            for ii in range(options['num_NNs']):
                env_list[ii].ch = ch_group[assignment_record[ii]]

            print("Assignment uses %s seconds." % (time.time() - start_time))

            # ---------- Learning ---------- #
            for beam_id in range(options['num_NNs']):
                train_opt_list[beam_id] = train(actor_net_list[beam_id],
                                                critic_net_list[beam_id],
                                                actor_net_t_list[beam_id],
                                                critic_net_t_list[beam_id],
                                                ounoise_list[beam_id],
                                                env_list[beam_id],
                                                options,
                                                train_opt_list[beam_id],
                                                beam_id)
                # print("Beam: ", beam_id, "Iter: ", loop_id)
            # print("--- %s seconds ---" % (time.time() - start_time))
