import numpy as np
import h5py as h5


def dataPrep(inputName=None):
    with h5.File(inputName, 'r') as f:
        fields = [k for k in f.keys()]
        nested = [k for k in f[fields[0]]]
        data_channels = np.squeeze(np.array(nested))
        decoup = data_channels.view(np.float64).reshape(data_channels.shape + (2,))
        X_real = decoup[:, :, 0]
        X_imag = decoup[:, :, 1]
        X = np.concatenate((X_real, X_imag), axis=-1)

    return X


# --- test script ---
# path = './CT_O1_3p5_BS3_5Paths_norm.mat'
# data = dataPrep(path)
