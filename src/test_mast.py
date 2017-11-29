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
from dataLoaders.MAST_Dataset import MAST_Dataset
from models.PitchContourAssessor import PitchContourAssessor
from sklearn import metrics

# check is cuda is available and print result
CUDA_AVAILABLE = torch.cuda.is_available()
print('Running on GPU: ', CUDA_AVAILABLE)

# define model
METRIC = 0
perf_model = PitchContourAssessor()
if CUDA_AVAILABLE:
    perf_model.cuda()
criterion = nn.MSELoss()   

# read the model
filename = '1400_5000_middle_0_Reg_2'
if torch.cuda.is_available():
    perf_model.cuda()
    perf_model.load_state_dict(torch.load('saved/' + filename + '.pt'))
else:
    perf_model.load_state_dict(torch.load('saved/' + filename + '.pt', map_location=lambda storage, loc: storage))

# import data from MAST dataset
if sys.version_info[0] < 3:
    mast_path = '/Users/Som/GitHub/Mastmelody_dataset/f0data'
else:
    mast_path = '/home/apati/Mastmelody_dataset/f0data'
mast_dataset = MAST_Dataset(mast_path)
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
    pitch_tensor = torch.from_numpy(f0).view(1,-1).float()
    if pitch_tensor.size(1) < 4000:
        pitch_tensor = torch.cat((pitch_tensor, torch.zeros(1, 4000 - pitch_tensor.size(1))), 1)
    d['pitch_tensor'] = pitch_tensor
    d['score_tensor'] = torch.from_numpy(np.ones((1,4)) * target).float()
    mast_data.append(d)    

# define evaluation methods
def eval_regression(target, pred):
    """
    Calculates the standard regression metrics
    Args:
        target:     (N x 1) torch Float tensor, actual ground truth
        pred:       (N x 1) torch Float tensor, predicted values from the regression model
    """
    if torch.cuda.is_available():
        pred_np = pred.clone().cpu().numpy()
        target_np = target.clone().cpu().numpy()
    else:
        pred_np = pred.clone().numpy()
        target_np = target.clone().numpy()
    
    # compute r-sq score 
    r_sq = metrics.r2_score(target_np, pred_np)
    # compute classification accuracy
    pred_class = np.rint(pred_np * 10)
    pred_class[pred_class < 0] = 0
    pred_class[pred_class > 10] = 10
    target_class = np.rint(target_np * 10)
    pred_class.astype(int)
    target_class.astype(int)
    accuracy = metrics.accuracy_score(target_class, pred_class, normalize=True)
    return r_sq, accuracy

def eval_classification(target, pred):
    """
    Calculates the binary classification accuracy for the regression model
    Args:
        target:     (N x 1) torch Float tensor, actual ground truth
        pred:       (N x 1) torch Float tensor, predicted values from the regression model
    """
    if torch.cuda.is_available():
        pred_class = pred.clone().cpu().numpy()
        target_class = target.clone().cpu().numpy()
    else:
        pred_class = pred.clone().numpy()
        target_class = target.clone().numpy()
    print(pred_class)
    print(target_class)
    pred_class[pred_class < 0.5] = 0
    pred_class[pred_class >= 0.5] = 1
    target_class[target_class < 0.5] = 0
    target_class[target_class >= 0.5] = 1
    pred_class.astype(int)
    target_class.astype(int)
    accuracy = metrics.accuracy_score(target_class, pred_class, normalize=True)
    return accuracy

# define evaluation method
def eval_model(model, data, metric):
    """
    Returns the model performance metrics
    Args:
        model:          object, trained model of PitchContourAssessor class
        data:           list, batched testing data
        metric:         int, from 0 to 3, which metric to evaluate against
    """
    # put the model in eval mode
    model.eval()
    # intialize variables
    num_batches = len(data)
    pred = np.array([])
    target = np.array([])
    loss_avg = 0
    # iterate over batches for validation
    for batch_idx in range(num_batches):
        # extract pitch tensor and score for the batch
        pitch_tensor = data[batch_idx]['pitch_tensor']
        score_tensor = data[batch_idx]['score_tensor'][:, metric]
        # prepare data for input to model
        model_input = pitch_tensor.clone()
        model_target = score_tensor.clone()
        # convert to cuda tensors if cuda available
        if CUDA_AVAILABLE:
            model_input = model_input.cuda()
            model_target = model_target.cuda()
        # wrap all tensors in pytorch Variable
        model_input = Variable(model_input)
        model_target = Variable(model_target)
        # compute forward pass for the network
        mini_batch_size = model_input.size(0)
        model.init_hidden(mini_batch_size)
        model_output = model(model_input)
        # compute loss
        loss = criterion(model_output, model_target)
        loss_avg += loss.data[0]
        # concatenate target and pred for computing validation metrics
        pred = torch.cat((pred, model_output.data.view(-1)), 0) if pred.size else model_output.data.view(-1)
        target = torch.cat((target, score_tensor), 0) if target.size else score_tensor
    r_sq, accuracy = eval_regression(target, pred)
    bin_accu = eval_classification(target, pred)
    loss_avg /= num_batches
    return loss_avg, r_sq, accuracy, bin_accu

# evaluate model on MAST dataset
test_loss, test_r_sq, test_accu, bin_accu = eval_model(perf_model, mast_data, METRIC)
print('[%s %0.5f]'% ('MAST Accuracy: ', bin_accu))
