import os
import torch
import dill
import numpy as np
from torch.utils.data import Dataset, DataLoader


class PitchContourDataset(Dataset):
    """Dataset class for pitch contour data"""
    
    def __init__(self, data_path):
        """
        Initializes the class, defines the number of datapoints
        Args: 
            data_path:  full path to the file which contains the pitch contour data
        """
        

    def __getitem__(self, idx):
        """ 
        Returns a datapoint (interval and duration indices) and corresponding targets for a particular index
        Args:
            idx:        int, must range within [0, length of dataset)
        """
        

    def __len__(self):
        """
        Return the size of the dataset
        """
        return self.length
    
    

class ZeroPad(object):
    """
    Adds stop tags to the datapoints in the 
    """
    def __init__(self, seq_length):
        """
        Initializes the ZeroPad class
        Args:
            seq_length:     int, length of the final zero padded sequence
        """
        assert isinstance(seq_length, int)
        self.seq_length = seq_length

    def apply_pad(self, sample):
        """
        Pads the input 1-D tensor so that it become the same length as the seq_length member of the class
        Args:
            sample: 1-D long tensor
        """
        assert self.seq_length >= sample.size(0)
        if self.seq_length == sample.size(0):
            return sample
        zero_pad = torch.zeros(self.seq_length - int(sample.size(0))).long()
        zero_padded_sample = torch.cat((sample, zero_pad), 0)
        return zero_padded_sample









