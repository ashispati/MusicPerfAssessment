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
from models.PCConvNet import PCConvNet
from models.PCConvLstmNet import PCConvLstmNet
from dataLoaders.PitchContourDataset import PitchContourDataset
from dataLoaders.PitchContourDataloader import PitchContourDataloader
from sklearn import metrics
import eval_utils

np.set_printoptions(precision=4)

# check is cuda is available and print result
CUDA_AVAILABLE = torch.cuda.is_available()
print('Running on GPU: ', CUDA_AVAILABLE)
if CUDA_AVAILABLE != True:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

# initializa training parameters
RUN = 15
NUM_EPOCHS = 4000
NUM_DATA_POINTS = 1410
NUM_BATCHES = 10
BAND = 'middle'
SEGMENT = '2'
METRIC = 0 # 0: Musicality, 1: Note Accuracy, 2: Rhythmic Accuracy, 3: Tone Quality
MTYPE = 'conv'

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

# initialize dataset, dataloader and created batched data
file_name = BAND + '_' + str(SEGMENT) + '_data'
if sys.version_info[0] < 3:
    data_path = 'dat/' + file_name + '.dill'
else:
    data_path = 'dat/' + file_name + '_3.dill'
dataset = PitchContourDataset(data_path)
dataloader = PitchContourDataloader(dataset, NUM_DATA_POINTS, NUM_BATCHES)
_, _, vef, _, tef = dataloader.create_split_data(1000, 500)
# test on full length data
test_loss, test_r_sq, test_accu, test_accu2, pred, target = eval_utils.eval_model(perf_model, criterion, tef, METRIC, MTYPE, 1)
print('[%s %0.5f, %s %0.5f, %s %0.5f %0.5f]'% ('Testing Loss: ', test_loss, ' R-sq: ', test_r_sq, ' Accu:', test_accu, test_accu2))

# convert to numpy
if torch.cuda.is_available():
    pred = pred.clone().cpu().numpy()
    target = target.clone().cpu().numpy()
else:
    pred = pred.clone().numpy()
    target = target.clone().numpy()
a = np.absolute((pred - target))

# compute correlation coefficient
R, p = ss.pearsonr(pred, target)
print(R, p)

# sort based ascending order of prediction residual
sort_ixs = np.argsort(a)
print(target)
print(a)
data_point = tef[0]
X = data_point['pitch_tensor']
y = data_point['score_tensor'][:, METRIC]
smap = np.absolute(eval_utils.compute_saliency_maps(X, y, perf_model).view(-1).numpy())
thres = np.median(smap)
X_np = np.around(X.view(-1).numpy(), decimals=3)
X_np[X_np == 0.0] = float('nan')
X_map = X_np.copy()
X_map[smap < thres] = float('nan')
plt.plot(X_np)
#plt.plot(X_map)
plt.ylabel('Pitch Contour')
plt.show()





