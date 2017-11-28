import os
import sys
import math
import time
import scipy.stats as ss
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn import metrics

"""
Contains standard utility functions for training and testing evaluations
"""

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
    pred_np[pred_np < 0] = 0
    pred_np[pred_np > 1] = 1
    print(pred_np)
    print(target_np)
    a = np.absolute((pred_np - target_np))
    r_sq = metrics.r2_score(target_np, pred_np)
    # compute 11-class classification accuracy
    pred_class = np.rint(pred_np * 10)
    pred_class[pred_class < 0] = 0
    pred_class[pred_class > 10] = 10
    target_class = np.rint(target_np * 10)
    pred_class.astype(int)
    target_class.astype(int)
    accu = metrics.accuracy_score(target_class, pred_class, normalize=True) 
    pred_class[pred_class < 5] = 0
    pred_class[pred_class >= 5] = 1
    target_class[target_class < 5] = 0
    target_class[target_class >= 5] = 1
    #print(target_class)
    accu2 = metrics.accuracy_score(target_class, pred_class, normalize=True)
    return r_sq, accu, accu2

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
        if torch.cuda.is_available():
            model_input = model_input.cuda()
            model_target = model_target.cuda()
        # wrap all tensors in pytorch Variable
        model_input = Variable(model_input)
        model_target = Variable(model_target)
        # compute forward pass for the network
        mini_batch_size = model_input.size(0)
        #model.init_hidden(mini_batch_size)
        model_output = model(model_input)
        # compute loss
        criterion = nn.MSELoss()
        loss = criterion(model_output, model_target)
        loss_avg += loss.data[0]
        # concatenate target and pred for computing validation metrics
        pred = torch.cat((pred, model_output.data.view(-1)), 0) if pred.size else model_output.data.view(-1)
        target = torch.cat((target, score_tensor), 0) if target.size else score_tensor
    r_sq, accu, accu2 = eval_regression(target, pred)
    loss_avg /= num_batches
    return loss_avg, r_sq, accu, accu2

def compute_saliency_maps(X, y, model):
    """
    Computes a regression score saliency map using the model for pitch contour X and score y.
    Args:
        X: Input pitch contour, torch tensor of shape (N, W)
        y: Regression score for X, Float tensor of shape (N,)
        model: A pretrained model that will be used to compute the saliency map.
    """
    # Set the model is in "test" mode
    model.eval()
    
    # Wrap the input tensors in Variables
    X_var = Variable(X, requires_grad=True)
    y_var = Variable(y, requires_grad=False)
    saliency = None
    
    # compute forward pass and class scores
    pred_scores = model.forward(X_var)
    criterion = nn.MSELoss()
    loss = criterion(pred_scores, y_var)

    # compute gradient wrt input
    loss.backward()
    saliency = X_var.grad
    
    saliency = torch.max(saliency, 1)[0].data
    return saliency

