import os
import dill
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class PCDataset(Dataset):
    """Dataset class for spectral feature based music performance assessment data"""

    def __init__(self, data_path, label_id):
        """
        Initializes the class, defines the number of datapoints
        Args:
            data_path:  full path to the file which contains the pitch contour data
            label_id:   the label to use for training
            rep_params: parameters for spectral representation
        """
        super(PCDataset, self).__init__()
        self.perf_data = dill.load(open(data_path, 'rb'))
        self.label_id = label_id
        self.length = len(self.perf_data)
            
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        label = self.perf_data[idx]['ratings'][self.label_id]
        # For one-hot labels
        # y = np.zeros(11)
        # y[label] = 1

        # For class labels
        y = label
        # return torch.unsqueeze(X, 0), y

        # For combined model
        pc = torch.FloatTensor(normalize_pitch_contour(self.perf_data[idx]['pitch_contour']))
        return pc, y


def normalize_pitch_contour(pitch_contour):
    """
    Returns the normalized pitch contour after converting to floating point MIDI
    Args:
        pitch_contour:      np 1-D array, contains pitch in Hz
    """
    # convert to MIDI first
    a4 = 440.0
    pitch_contour[pitch_contour != 0] = 69 + 12 * \
        np.log2(pitch_contour[pitch_contour != 0] / a4)
    # normalize pitch (restrict between 36 to 108 MIDI notes)
    normalized_pitch = pitch_contour #/ 127.0
    normalized_pitch[normalized_pitch != 0] = (normalized_pitch[normalized_pitch != 0] - 36.0)/72.0 
    return normalized_pitch


def _collate_fn(batch):
    def func_pc(p):
        return p[0].size(0)
    
    longest_sample_pc = max(batch, key=func_pc)[0]
    minibatch_size = len(batch)
    max_pclength = longest_sample_pc.size(0)
    inputs_pc = torch.zeros(minibatch_size, max_pclength)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        pc = sample[0]
        target = sample[1]
        pc_length = pc.size(0)
        inputs_pc[x].narrow(0, 0, pc_length).copy_(pc)
        targets.append(target)
    targets = torch.Tensor(targets)
    return inputs_pc, targets


class PCDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super(PCDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn