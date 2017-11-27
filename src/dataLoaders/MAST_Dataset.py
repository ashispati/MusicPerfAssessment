import os
import torch
from torch.utils.data import Dataset
import numpy as np

class MAST_Dataset(Dataset):
    # f0_path is the path to the f0data directory in MAST_dataset
    def __init__(self, f0_path):
        super(MAST_Dataset, self).__init__()
        all_f0s = os.listdir(f0_path)
        self.data = [(a,1) for a in all_f0s if 'pass' in a] # 266 samples
        self.data.extend([(a,0) for a in all_f0s if 'fail' in a]) # 730 samples
        self.f0_path = f0_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file, target = self.data[index]
        f0 = np.loadtxt(os.path.join(self.f0_path, file))[:,1]
        # downsample
        f0 = f0[0::2]
        return f0, target
