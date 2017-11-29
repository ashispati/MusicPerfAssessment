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
from models.PCConvNet import PCConvNet
from models.PCConvLstmNet import PCConvLstmNet
from dataLoaders.PitchContourDataset import PitchContourDataset
from dataLoaders.PitchContourDataloader import PitchContourDataloader
from tensorboard_logger import configure, log_value
from sklearn import metrics
import eval_utils
import train_utils

# set manual random seed for reproducibility
torch.manual_seed(1)
np.random.seed(1)
# check is cuda is available and print result
CUDA_AVAILABLE = torch.cuda.is_available()
print('Running on GPU: ', CUDA_AVAILABLE)
if CUDA_AVAILABLE != True:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

# initializa training parameters
RUN = 21
NUM_EPOCHS = 2000
NUM_DATA_POINTS = 1410
NUM_BATCHES = 10
BAND = 'middle'
SEGMENT = '2'
METRIC = 0 # 0: Musicality, 1: Note Accuracy, 2: Rhythmic Accuracy, 3: Tone Quality
MTYPE = 'lstm'
# initialize dataset, dataloader and created batched data
file_name = BAND + '_' + str(SEGMENT) + '_data'
if sys.version_info[0] < 3:
    data_path = 'dat/' + file_name + '.dill'
else:
    data_path = 'dat/' + file_name + '_3.dill'
dataset = PitchContourDataset(data_path)
dataloader = PitchContourDataloader(dataset, NUM_DATA_POINTS, NUM_BATCHES)
tr1, v1, _, te1, _ = dataloader.create_split_data(1000, 500)
tr2, v2, _, te2, _ = dataloader.create_split_data(1500, 500)
tr3, v3, _, te3, _ = dataloader.create_split_data(2000, 1000)
tr4, v4, _, te4, _ = dataloader.create_split_data(2500, 1000)
tr5, v5, _, te5, _ = dataloader.create_split_data(3000, 1500)
tr6, v6, vef, te6, tef = dataloader.create_split_data(4000, 2000)
training_data = tr1 + tr2 + tr3 + tr4 + tr5 + tr6
validation_data = vef #+ v2 + v3 + v4 + v5 + v6
#testing_data = te1 + te2 + te3 + te4 + te5 + te6

# augment data
aug_training_data = training_data#train_utils.augment_data(training_data)
#aug_training_data = train_utils.augment_data(aug_training_data)
aug_validation_data = validation_data  #train_utils.augment_data(validation_data)

## initialize model
if MTYPE == 'conv':
    perf_model = PCConvNet(0)
elif MTYPE == 'lstm':
    perf_model = PCConvLstmNet()        
if torch.cuda.is_available():
    perf_model.cuda()
criterion = nn.MSELoss()
LR_RATE = 0.01
W_DECAY = 1e-5
MOMENTUM = 0.9
perf_optimizer = optim.SGD(perf_model.parameters(), lr= LR_RATE, momentum=MOMENTUM, weight_decay=W_DECAY)
#perf_optimizer = optim.Adam(perf_model.parameters(), lr = LR_RATE, weight_decay = W_DECAY)
print(perf_model)

# declare save file name
file_info = str(NUM_DATA_POINTS) + '_' + str(NUM_EPOCHS) + '_' + BAND + '_' + str(METRIC) + '_' + str(RUN) + '_' + MTYPE        

# configure tensor-board logger
configure('runs/' + file_info + '_Reg' , flush_secs = 2)

## define training parameters
PRINT_EVERY = 1
ADJUST_EVERY = 1000
START = time.time()
best_val_loss = 1.0

# train and validate
try:
    print("Training for %d epochs..." % NUM_EPOCHS)
    for epoch in range(1, NUM_EPOCHS + 1):
        # perform training and validation
        train_loss, train_r_sq, train_accu, train_accu2, val_loss, val_r_sq, val_accu, val_accu2 = train_utils.train_and_validate(perf_model, criterion, perf_optimizer, aug_training_data, aug_validation_data, METRIC, MTYPE)
        # adjut learning rate
        # train_utils.adjust_learning_rate(perf_optimizer, epoch, ADJUST_EVERY)
        # log data for visualization later
        log_value('train_loss', train_loss, epoch)
        log_value('val_loss', val_loss, epoch)
        log_value('train_r_sq', train_r_sq, epoch)
        log_value('val_r_sq', val_r_sq, epoch)
        log_value('train_accu', train_accu, epoch)
        log_value('val_accu', val_accu, epoch)
        log_value('train_accu2', train_accu2, epoch)
        log_value('val_accu2', val_accu2, epoch)
        # print loss
        if epoch % PRINT_EVERY == 0:
            print('[%s (%d %.1f%%)]' % (train_utils.time_since(START), epoch, float(epoch) / NUM_EPOCHS * 100))
            print('[%s %0.5f, %s %0.5f, %s %0.5f %0.5f]'% ('Train Loss: ', train_loss, ' R-sq: ', train_r_sq, ' Accu:', train_accu, train_accu2))
            print('[%s %0.5f, %s %0.5f, %s %0.5f %0.5f]'% ('Valid Loss: ', val_loss, ' R-sq: ', val_r_sq, ' Accu:', val_accu, val_accu2))
        # save model if best validation loss
        if val_loss < best_val_loss:
            n = file_info + '_best'
            train_utils.save(n, perf_model)
            best_val_loss = val_loss
    print("Saving...")
    train_utils.save(file_info, perf_model)
except KeyboardInterrupt:
    print("Saving before quit...")
    train_utils.save(file_info, perf_model)

# test
# test of full length data
test_loss, test_r_sq, test_accu, test_accu2 = eval_utils.eval_model(perf_model, criterion, tef, METRIC, MTYPE)
print('[%s %0.5f, %s %0.5f, %s %0.5f %0.5f]'% ('Testing Loss: ', test_loss, ' R-sq: ', test_r_sq, ' Accu:', test_accu, test_accu2))

# validate and test on best validation model
# read the model
filename = file_info + '_best' + '_Reg'
if torch.cuda.is_available():
    perf_model.cuda()
    perf_model.load_state_dict(torch.load('saved/' + filename + '.pt'))
else:
    perf_model.load_state_dict(torch.load('saved/' + filename + '.pt', map_location=lambda storage, loc: storage))

val_loss, val_r_sq, val_accu, val_accu2 = eval_utils.eval_model(perf_model, criterion, vef, METRIC, MTYPE)
print('[%s %0.5f, %s %0.5f, %s %0.5f %0.5f]'% ('Valid Loss: ', val_loss, ' R-sq: ', val_r_sq, ' Accu:', val_accu, val_accu2))

test_loss, test_r_sq, test_accu, test_accu2 = eval_utils.eval_model(perf_model, criterion, tef, METRIC, MTYPE)
print('[%s %0.5f, %s %0.5f, %s %0.5f %0.5f]'% ('Testing Loss: ', test_loss, ' R-sq: ', test_r_sq, ' Accu:', test_accu, test_accu2))
