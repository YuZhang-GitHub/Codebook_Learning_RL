import numpy as np
import scipy.io as scio

num_ant = 32
num_beam = 4
results = np.empty((num_beam, 2*num_ant))

path = './beams/'

for beam_id in range(num_beam):
    fname = 'beams_' + str(beam_id) + '_max.txt'
    with open(path + fname, 'r') as f:
        lines = f.readlines()
        last_line = lines[-1]
        results[beam_id, :] = np.fromstring(last_line.replace("\n", ""), sep=',').reshape(1, -1)

results = (1 / np.sqrt(num_ant)) * (results[:, ::2] + 1j * results[:, 1::2])

scio.savemat('beam_codebook.mat', {'beams': results})
