from __future__ import print_function
import os
import dill
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class RawAudioDataset(Dataset):
    """Dataset class for pitch contour based music performance assessment data"""

    def __init__(self, data_path):
        """
        Initializes the class, defines the number of datapoints
        Args:
            data_path:  full path to the file which contains the pitch contour data
        """
        assert data_path.split('.')[1] == 'dill'
        self.data = dill.load(open(data_path, 'rb'))
        self.length = len(self.data)

        # perform a few pre-processing steps
        for i in range(self.length):
            self.data[i]['sampling_rate'] = self.data[i]['audio'][1]
            self.data[i]['audio'] = self._normalize(self.data[i]['audio'][0])
            self.data[i]['length'] = len(self.data[i]['audio'])
        return

    def __getitem__(self, idx):
        """
        Returns a datapoint for a particular index
        Args:
            idx:    int, must range within [0, length of dataset)
        """
        return self.data[idx]

    def __len__(self):
        """
        Return the size of the dataset
        """
        return self.length

    def _normalize(self, sequence):
        """
        Returns the sequence normalized between -1 and 1
        Args:
            sequence:   np 1-D float array
        """
        maxval = np.max(sequence)
        return sequence/float(maxval)

    def plot_audio(self, idx):
        """
        Plots the audio for visualization
        """
        audio = self.data[idx]['audio']
        plt.plot(audio)
        plt.ylabel('pYin Audio (in Hz)')
        plt.show()

