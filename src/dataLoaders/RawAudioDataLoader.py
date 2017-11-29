from __future__ import print_function
import os
import sys
import collections
import numpy as np
from random import shuffle
import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as multiprocessing
from dataLoaders.RawAudioDataset import RawAudioDataset

# set manual random seed for reproducibility
torch.manual_seed(1)
np.random.seed(1)
class RawAudioDataLoader(DataLoader):
    """
    DataLoader class for raw audio music performance assessment data
    """

    def __init__(self, dataset, num_data_pts=None, num_batches=1):
        """
        Initializes the class, defines the number of batches and other parameters
        Args:
                dataset:  	object of the RawAudioDataset class, should be properly initialized
                num_data_pts:	int, number of data points to be considered while loading the data
                num_batches:	int, number of batches to be considered
        """
        if num_data_pts is None:
            num_data_pts = len(dataset)
        # check if input parameters are accurate
        assert num_data_pts <= len(dataset)
        assert num_batches <= num_data_pts
        self.dataset = dataset
        self.num_data_pts = num_data_pts
        self.num_batches = num_batches
        self.mini_batch_size = int(np.floor(self.num_data_pts / self.num_batches))

    def _zero_pad(self, sequence, length):
        """
        Zero-pads the input 1-D float tensor so that it is as long as 'length'
        Args:
            sequence: 1-D double tensor
            length: desired length of zero padded sequence
        """
        assert length >= len(sequence)
        if length == len(sequence):
            return sequence
        zero_padding = torch.zeros(length - int(sequence.size(0)))
        padded_sequence = torch.cat((sequence, zero_padding), 0)
        return padded_sequence

    def _get_sorted_data(self):
        """
        Returns data points sorted in descending order of raw audio length
        """
        # get the lengths of all data points
        audio_lengths = np.empty([self.num_data_pts])
        for i in range(self.num_data_pts):
            audio_lengths[i] = len(self.dataset[i]['audio'])

        # get the sorted indices
        sorted_indices = (-audio_lengths).argsort(kind='mergesort')

        # iterate and collect data
        sorted_data = self.dataset[sorted_indices]
        return sorted_data

    def create_batched_data(self):
        """
        Returns batched data after sorting as a list
        """
        # sort the data first
        sorted_data = self._get_sorted_data()
        # batch the sorted data
        count = 0
        batched_data = []
        for batch_num in range(self.num_batches):
            longest_audio_length = len(sorted_data[count]['audio'])
            audio_tensor = torch.zeros(self.mini_batch_size, longest_audio_length)
            ratings_tensor = torch.zeros(self.mini_batch_size, len(sorted_data[count]['ratings']))
            class_ratings_tensor = torch.zeros(self.mini_batch_size,
                                               len(sorted_data[count]['class_ratings'])).long()
            for i in range(self.mini_batch_size):
                # convert desired data to torch tensors
                audio_tensor[i, :] = self._zero_pad(torch.FloatTensor(sorted_data[count]['audio']),
                                                    longest_audio_length)
                ratings_tensor[i, :] = torch.FloatTensor(sorted_data[count]['ratings'])
                class_ratings_tensor[i, :] = torch.LongTensor(sorted_data[count]['class_ratings'])
                count += 1
            data = {}
            data['audio_tensor'] = audio_tensor
            data['ratings_tensor'] = ratings_tensor
            data['class_ratings_tensor'] = class_ratings_tensor
            batched_data.append(data)
        print(count)
        return batched_data

    # def create_split_data(self, chunk_len, hop):
    #     """
    #     Returns batched data which is split into chunks
    #     Args:
    #         chunk_len:  length of the chunk in samples
    #         hop:	hop length in samples
    #     """
    #     indices = np.arange(self.num_data_pts)
    #     np.random.shuffle(indices)
    #     num_training_songs = int(0.8 * self.num_data_pts)
    #     num_validation_songs = int(0.1 * self.num_data_pts)
    #     num_testing_songs = int(0.1 * self.num_data_pts)
    #     train_split = []
    #     for i in range(num_training_songs):
    #         data = self.dataset[indices[i]]
    #         pc = data['pitch_contour']
    #         gt = data['ratings']
    #         count = 0
    #         if len(pc) < chunk_len:
    #             zeropad_pc = np.zeros((chunk_len,))
    #             zeropad_pc[:pc.shape[0],] = pc
    #             pc = zeropad_pc
    #         while count + chunk_len < len(pc):
    #             d = {}
    #             d['pitch_contour'] = pc[count: count+chunk_len]
    #             d['ratings'] = gt
    #             train_split.append(d)
    #             count += hop
    #     shuffle(train_split)
    #     num_data_pts = len(train_split)
    #     batched_data = [None] * self.num_batches
    #     mini_batch_size = int(np.floor(num_data_pts / self.num_batches))
    #     count = 0
    #     for batch_num in range(self.num_batches):
    #         batched_data[batch_num] = list()
    #         pitch_tensor = torch.zeros(mini_batch_size, chunk_len)
    #         score_tensor = torch.zeros(mini_batch_size, len(train_split[count]['ratings']))
    #         for seq_num in range(mini_batch_size):
    #             # convert pitch contour to torch tensor
    #             pc_tensor = torch.from_numpy(train_split[count]['pitch_contour'])
    #             pitch_tensor[seq_num, :] = pc_tensor.float()
    #             # convert score tuple to torch tensor
    #             s_tensor = torch.from_numpy(np.asarray(train_split[count]['ratings']))
    #             score_tensor[seq_num, :] = s_tensor
    #             count += 1
    #         dummy = {}
    #         dummy['pitch_tensor'] = pitch_tensor
    #         dummy['score_tensor'] = score_tensor
    #         batched_data[batch_num] = dummy

    #     val_split = []
    #     for i in range(num_training_songs, num_training_songs + num_validation_songs):
    #         data = self.dataset.__getitem__(indices[i])
    #         pc = data['pitch_contour']
    #         gt = data['ratings']
    #         count = 0
    #         if len(pc) < chunk_len:
    #             zeropad_pc = np.zeros((chunk_len,))
    #             zeropad_pc[:pc.shape[0],] = pc
    #             pc = zeropad_pc
    #         while count + chunk_len < len(pc):
    #             d = {}
    #             d['pitch_contour'] = pc[count: count+chunk_len]
    #             d['ratings'] = gt
    #             val_split.append(d)
    #             count += hop
    #     shuffle(val_split)
    #     num_data_pts = len(val_split)
    #     mini_batch_size = num_data_pts
    #     count = 0
    #     pitch_tensor = torch.zeros(mini_batch_size, chunk_len)
    #     score_tensor = torch.zeros(mini_batch_size, len(val_split[count]['ratings']))
    #     for seq_num in range(mini_batch_size):
    #         # convert pitch contour to torch tensor
    #         pc_tensor = torch.from_numpy(val_split[count]['pitch_contour'])
    #         pitch_tensor[seq_num, :] = pc_tensor.float()
    #         # convert score tuple to torch tensor
    #         s_tensor = torch.from_numpy(np.asarray(val_split[count]['ratings']))
    #         score_tensor[seq_num, :] = s_tensor
    #         count += 1
    #     dummy = {}
    #     dummy['pitch_tensor'] = pitch_tensor
    #     dummy['score_tensor'] = score_tensor
    #     val_batch = [dummy]

    #     test_split = []
    #     for i in range(num_training_songs + num_validation_songs, num_training_songs + num_validation_songs + num_testing_songs):
    #         data = self.dataset.__getitem__(indices[i])
    #         pc = data['pitch_contour']
    #         gt = data['ratings']
    #         count = 0
    #         if len(pc) < chunk_len:
    #             zeropad_pc = np.zeros((chunk_len,))
    #             zeropad_pc[:pc.shape[0],] = pc
    #             pc = zeropad_pc
    #         while count + chunk_len < len(pc):
    #             d = {}
    #             d['pitch_contour'] = pc[count: count+chunk_len]
    #             d['ratings'] = gt
    #             test_split.append(d)
    #             count += hop
    #     num_data_pts = len(test_split)
    #     mini_batch_size = num_data_pts
    #     count = 0
    #     pitch_tensor = torch.zeros(mini_batch_size, chunk_len)
    #     score_tensor = torch.zeros(mini_batch_size, len(test_split[count]['ratings']))
    #     for seq_num in range(mini_batch_size):
    #         # convert pitch contour to torch tensor
    #         pc_tensor = torch.from_numpy(test_split[count]['pitch_contour'])
    #         pitch_tensor[seq_num, :] = pc_tensor.float()
    #         # convert score tuple to torch tensor
    #         s_tensor = torch.from_numpy(np.asarray(test_split[count]['ratings']))
    #         score_tensor[seq_num, :] = s_tensor
    #         count += 1
    #     dummy = {}
    #     dummy['pitch_tensor'] = pitch_tensor
    #     dummy['score_tensor'] = score_tensor
    #     test_batch = [dummy]
    #     return batched_data, val_batch, test_batch
