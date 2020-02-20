from __future__ import print_function
import gc
import os
import sys
import math
import time
import scipy.stats as ss
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.PCConvNet import PCConvNet, PCConvNetCls
from models.PCConvLstmNet import PCConvLstmNet, PCConvLstmNetCls
from dataLoaders.PitchContourDataset import PitchContourDataset
from dataLoaders.PitchContourDataloader import PitchContourDataloader
from dataLoaders.MASTDataset import MASTDataset
from dataLoaders.MASTDataloader import MASTDataloader
#from tensorboard_logger import configure, log_value
from sklearn import metrics
import eval_utils
import train_utils

#DEFINE MODEL
perf_model = PCConvNet(0)

#SET ATTRIBUTES AND FILENAME - see script_train.py for what these should look like
NUM_EPOCHS = 2000
NUM_BATCHES = 1
BAND = 'symphonic'
METRIC = 0
#RUN = 13
RUN = 110
#RUN = 13
MTYPE = 'conv'
ctype = 0	#Set to 0 for regression, 1 for classification

if BAND == 'middle':
	NUM_DATA_POINTS = 1410
elif BAND == 'symphonic':
	NUM_DATA_POINTS = 1550
else:
	NUM_DATA_POINTS = 1410

# Make sure this filename will be correct -- should already be consistent with naming conventions from script_train.py
filename = str(NUM_DATA_POINTS) + '_' + str(NUM_EPOCHS) + '_' + BAND + '_' + str(METRIC) + '_' + str(RUN) + '_' + MTYPE + '_best'

## LOAD SAVED MODEL
# change the string argument to reflect wherever your 'saved_runs' folder is located
perf_model.load_state_dict(torch.load('/Users/michaelfarren/Desktop/MusicTech/MPA_new/saved_runs/' + filename, map_location=lambda storage, loc: storage)) 

#Enter eval mode
perf_model.eval()

#SET PATH TO .DILL DATA
data_path = '/Volumes/Farren/Python_stuff/dat/' + BAND + '_2_data_3.dill'

print('Setting up data...')
dataset = PitchContourDataset(data_path)
print("Size of train_" + BAND + ": " + str(len(dataset)))
#error-catching for now
#NUM_DATA_POINTS = 155
dataloader = PitchContourDataloader(dataset, NUM_DATA_POINTS, NUM_BATCHES)

tr1, v1, vef, te1, tef = dataloader.create_split_data(1000, 500)
data = tef
print("Size of one pitch contour input: " + str(len(data[0]['pitch_tensor'])))

newDict = {}
newDict['pitch_list'] = data[0]['pitch_tensor'].tolist()
newDict['score_list'] = data[0]['score_tensor'].tolist()


#SET BATCH INDEX (temporary) - max value is len(data)
batch_idx = 13

# extract pitch tensor and score for the batch
pitch_tensor = data[batch_idx]['pitch_tensor']
score_tensor = data[batch_idx]['score_tensor'][:, METRIC]
# prepare data for input to model
model_input = pitch_tensor.clone()
model_target = score_tensor.clone()
if ctype == 1:
    model_input = model_input.long()
    model_target = model_target.long()
# convert to cuda tensors if cuda available
if torch.cuda.is_available():
    model_input = model_input.cuda()
    model_target = model_target.cuda()
# wrap all tensors in pytorch Variable
model_input = Variable(model_input)
model_target = Variable(model_target)
# compute forward pass for the network
mini_batch_size = model_input.size(0)
if MTYPE == 'lstm':
    model.init_hidden(mini_batch_size)
model_output = perf_model(model_input)

print(model_output.data)
print(model_target.data)

out = model_output.data.tolist()[0][0]
truth = model_target.data.tolist()[0]
print("Output: " + str(out) + ", True assessment score: " + str(truth))


#See how a loaded model performs overall

criterion = nn.MSELoss()

val_loss, val_r_sq, val_accu, val_accu2 = eval_utils.eval_model(perf_model, criterion, vef, METRIC, MTYPE, ctype)
print('[%s %0.5f, %s %0.5f, %s %0.5f %0.5f]'% ('Valid Loss: ', val_loss, ' R-sq: ', val_r_sq, ' Accu:', val_accu, val_accu2))

test_loss, test_r_sq, test_accu, test_accu2 = eval_utils.eval_model(perf_model, criterion, tef, METRIC, MTYPE, ctype)
print('[%s %0.5f, %s %0.5f, %s %0.5f %0.5f]'% ('Testing Loss: ', test_loss, ' R-sq: ', test_r_sq, ' Accu:', test_accu, test_accu2))





