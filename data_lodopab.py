import numpy as np
import torch
import os
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class LoDoPab_train(Dataset):
    def __init__(self, ground_truth_path, low_dose_path):
        self.ground_truth = []
        self.low_dose = []

        n = 40
        for i in range(n):
            with h5py.File(os.path.join(
                ground_truth_path, 'ground_truth_{}_{:03d}.hdf5'.format('train', i)), 'r') as file:
                ground_truth_data_i = file['data'][:]
            self.ground_truth.append(ground_truth_data_i)

            low_dose_data_i = np.load(os.path.join(low_dose_path, 'low_dose_{:03d}.npy'.format(i)))
            self.low_dose.append(low_dose_data_i)
                
        # ground_truth_data_i # [128, 362, 362]
        self.ground_truth = torch.FloatTensor(self.ground_truth).reshape((n * 128, 1, 362, 362))
        self.low_dose = torch.FloatTensor(self.low_dose).reshape((n * 128, 1, 362, 362))


    def __len__(self):
        return self.ground_truth.shape[0]

    def __getitem__(self, index):
        return self.ground_truth[index], self.low_dose[index]


class LoDoPab_test(Dataset):
    def __init__(self, ground_truth_path, low_dose_path):
        self.ground_truth = []
        self.low_dose = []

        n = 50
        p = 1
        for i in range(n, n+p):
            with h5py.File(os.path.join(
                ground_truth_path, 'ground_truth_{}_{:03d}.hdf5'.format('train', i)), 'r') as file:
                ground_truth_data_i = file['data'][:]
            self.ground_truth.append(ground_truth_data_i)

            low_dose_data_i = np.load(os.path.join(low_dose_path, 'low_dose_{:03d}.npy'.format(i)))
            self.low_dose.append(low_dose_data_i)
                
        # ground_truth_data_i # [128, 362, 362]
        self.ground_truth = torch.FloatTensor(self.ground_truth).reshape((p * 128, 1, 362, 362))
        self.low_dose = torch.FloatTensor(self.low_dose).reshape((p * 128, 1, 362, 362))


    def __len__(self):
        return self.ground_truth.shape[0]

    def __getitem__(self, index):
        return self.ground_truth[index], self.low_dose[index]
