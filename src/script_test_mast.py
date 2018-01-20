from __future__ import print_function
import gc
import os
import sys
import math
import scipy.stats as ss
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from dataLoaders.MASTDataset import MASTDataset
from dataLoaders.MASTDataloader import MASTDataloader
from models.PCConvNet import PCConvNet
from models.PCConvLstmNet import PCConvLstmNet
from eval_utils import eval_model, eval_regression
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

# check is cuda is available and print result
CUDA_AVAILABLE = torch.cuda.is_available()
print('Running on GPU: ', CUDA_AVAILABLE)

# define model

RUN = 18
NUM_EPOCHS = 2000
NUM_DATA_POINTS = 1410
BAND = 'middle'
SEGMENT = '2'
METRIC = 0
MTYPE = 'lstm'
# define model
if MTYPE == 'conv':
    perf_model = PCConvNet(0)
elif MTYPE == 'lstm':
    perf_model = PCConvLstmNet()
if CUDA_AVAILABLE:
    perf_model.cuda()
criterion = nn.MSELoss() 
 
# read the model
filename = str(NUM_DATA_POINTS) + '_' + str(NUM_EPOCHS) + '_' + BAND + '_' + str(METRIC) + '_' + str(RUN) + '_' + MTYPE + '_best_Reg'
if torch.cuda.is_available():
    perf_model.cuda()
    perf_model.load_state_dict(torch.load('saved/' + filename + '.pt'))
else:
    perf_model.load_state_dict(torch.load('saved/' + filename + '.pt', map_location=lambda storage, loc: storage))

def normalize_pitch_contour(pitch_contour):
    """
    Returns the normalized pitch contour after converting to floating point MIDI
    Args:
        pitch_contour:      np 1-D array, contains pitch in Hz
    """
    # convert to MIDI first
    # convert to MIDI first
    a4 = 440.0
    pitch_contour[pitch_contour != 0] = 69 + 12 * np.log2(pitch_contour[pitch_contour != 0] / a4)
    # normalize pitch (restrict between 36 to 108 MIDI notes)
    normalized_pitch = pitch_contour #/ 127.0
    normalized_pitch[normalized_pitch != 0] = (normalized_pitch[normalized_pitch != 0] - 36.0)/72.0 
    return normalized_pitch

def plot_pitch_contour(f0):
    """
    Plots the pitch contour for visualization
    """
    f0[f0 == 0.0] = float('nan')
    plt.plot(f0)
    plt.ylabel('pYin Pitch Contour (in Hz)')
    plt.show()

# import data from MAST dataset
if sys.version_info[0] < 3:
    mast_path = '/Users/Som/GitHub/Mastmelody_dataset/f0data'
else:
    mast_path = '/home/apati/MASTmelody_dataset/f0data'
mast_dataset = MASTDataset(mast_path)
mast_len = mast_dataset.__len__()
mast_data = []
fail_count = 0
for i in range(mast_len):
    f0, target = mast_dataset.__getitem__(i)
    if target == 0:
        fail_count += 1
    if fail_count > 266:
        continue
    d = {}
    pitch_tensor = torch.from_numpy(f0).float()
    pitch_tensor = pitch_tensor.view(1, -1)
    if pitch_tensor.size(1) < 1000:
        pitch_tensor = torch.cat((pitch_tensor, torch.zeros(1, 1000 - pitch_tensor.size(1))), 1)
    d['pitch_tensor'] = pitch_tensor
    d['score_tensor'] = torch.from_numpy(np.ones((1, 4)) * target).float()
    mast_data.append(d)

# evaluate model on MAST dataset
test_loss, test_r_sq, test_accu, bin_accu, pred, target = eval_model(perf_model, criterion, mast_data, METRIC, MTYPE, extra_outs=1)
print('[%s %0.5f]'% ('MAST Accuracy: ', bin_accu))


