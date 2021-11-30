import numpy as np
from DFT_gen import DFT_gen
from function_lib import phases_gen, bf_gain_cal, corr_mining, bf_gain_cal_np, corr_mining_np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier


def clustering_method(ch, num_cluster, mode, sensing_beam=None, num_bit=3, num_ant=16, num_random_beam=16):
    if mode == 'dft':
        dft_mtx = DFT_gen(mode)  # rows are DFT bf vectors
        user_vec_dft = bf_gain_cal(dft_mtx, ch)
        user_feature_dft = corr_mining(user_vec_dft)
        kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(np.transpose(user_feature_dft))
        return dft_mtx, kmeans
    elif mode == 'random':
        u_circ_pts = phases_gen(num_bit)
        F = np.zeros((num_random_beam, 2 * num_ant))
        for kk in range(num_random_beam):
            for jj in range(num_ant):
                F[kk, jj * 2:(jj + 1) * 2] = u_circ_pts[np.random.randint(0, 2 ** num_bit), :]
        user_vec = bf_gain_cal(F, ch)
        user_feature_vec = corr_mining(user_vec)
        kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(np.transpose(user_feature_vec))
        return F, kmeans
    elif mode == 'best':
        s_beam = sensing_beam
        user_vec_best = bf_gain_cal(s_beam, ch)
        user_feature_mtx = corr_mining(user_vec_best)
        kmeans = KMeans(n_clusters=num_cluster, random_state=0).fit(np.transpose(user_feature_mtx))
        return s_beam, kmeans
    else:
        ValueError('clustering mode is wrong.')


def kMeans_kNN(ch, n_cluster, n_bit=3, n_rand_beam=30):
    n_ant = int(ch.shape[1] / 2)
    u_circ_pts = phases_gen(n_bit)
    F = np.zeros((n_rand_beam, 2 * n_ant))
    for kk in range(n_rand_beam):
        for jj in range(n_ant):
            F[kk, jj * 2:(jj + 1) * 2] = u_circ_pts[np.random.randint(0, 2 ** n_bit), :]
    BFgain_matrix = bf_gain_cal_np(F, ch)
    F_matrix = corr_mining_np(BFgain_matrix)
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(np.transpose(F_matrix))
    target = kmeans.labels_
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(np.transpose(F_matrix), target)
    return neigh, F


def KMeans_only(ch, n_cluster, n_bit=3, n_rand_beam=30):
    n_ant = int(ch.shape[1] / 2)
    u_circ_pts = phases_gen(n_bit)
    F = np.zeros((n_rand_beam, 2 * n_ant))
    for kk in range(n_rand_beam):
        for jj in range(n_ant):
            F[kk, jj * 2:(jj + 1) * 2] = u_circ_pts[np.random.randint(0, 2 ** n_bit), :]
    BFgain_matrix = bf_gain_cal_np(F, ch)
    F_matrix = corr_mining_np(BFgain_matrix)
    kmeans = KMeans(n_clusters=n_cluster, random_state=0).fit(np.transpose(F_matrix))
    return kmeans, F
