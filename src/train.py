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
from models.PitchContourAssessor import PitchContourAssessor
from dataLoaders.PitchContourDataset import PitchContourDataset
from dataLoaders.PitchContourDataloader import PitchContourDataloader
from tensorboard_logger import configure, log_value
from sklearn import metrics

# set manual random seed for reproducibility
torch.manual_seed(1)

# check is cuda is available and print result
CUDA_AVAILABLE = torch.cuda.is_available()
print('Running on GPU: ', CUDA_AVAILABLE)
if CUDA_AVAILABLE != True:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

# initializa training parameters
NUM_EPOCHS = 10000
NUM_DATA_POINTS = 1400
NUM_BATCHES = 10
BAND = 'middle'
SEGMENT = '2'
METRIC = 0 # 0: Musicality, 1: Note Accuracy, 2: Rhythmic Accuracy, 3: Tone Quality

# initialize dataset, dataloader and created batched data
file_name = BAND + '_' + str(SEGMENT) + '_data'
if sys.version_info[0] < 3:
    data_path = 'dat/' + file_name + '.dill'
else:
    data_path = 'dat/' + file_name + '_3.dill'
dataset = PitchContourDataset(data_path)
dataloader = PitchContourDataloader(dataset, NUM_DATA_POINTS, NUM_BATCHES)
batched_data = dataloader.create_batched_data()

# split batches into training, validation and testing
training_data = batched_data[0:8]
validation_data = batched_data[8:9] 
testing_data = batched_data[9:10]

## initialize model
perf_model = PitchContourAssessor()
if CUDA_AVAILABLE:
    perf_model.cuda()
criterion = nn.MSELoss()
LR_RATE = 0.1
W_DECAY = 1e-5
perf_optimizer = optim.SGD(perf_model.parameters(), lr =  LR_RATE, weight_decay = W_DECAY)
print(perf_model)

# define evaluation method
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
    r_sq = metrics.r2_score(target_np, pred_np)
    return r_sq

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
    r_sq = eval_regression(target, pred)
    loss_avg /= num_batches
    return loss_avg, r_sq

def train(model, criterion, optimizer, data, metric):
    """
    Returns the model performance metrics
    Args:
        model:          object, trained model of PitchContourAssessor class
        criterion:      object, of torch.nn.Functional class which defines the loss 
        optimizer:      object, of torch.optim class which defines the optimization algorithm
        data:           list, batched testing data
        metric:         int, from 0 to 3, which metric to evaluate against
    """
    # Put the model in training mode
    model.train() 
    # Initializations
    num_batches = len(data)
    loss_avg = 0
	# iterate over batches for training
    for batch_idx in range(num_batches):
		# clear gradients and loss
        model.zero_grad()
        loss = 0

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
        # compute backward pass and step
        loss.backward()
        optimizer.step()
        # add loss
        loss_avg += loss.data[0]
    loss_avg /= num_batches
    return loss_avg

# define training method
def train_and_validate(model, criterion, optimizer, train_data, val_data, metric):
    """
    Defines the training and validation cycle for the input batched data
    Args:
        model:          object, trained model of PitchContourAssessor class
        criterion:      object, of torch.nn.Functional class which defines the loss 
        optimizer:      object, of torch.optim class which defines the optimization algorithm
        train_data:     list, batched training data
        val_data:       list, batched validation data
        metric:         int, from 0 to 3, which metric to evaluate against
    """
    # train the network
    train(model, criterion, optimizer, train_data, metric)
    # evaluate the network on train data
    train_loss_avg, train_r_sq = eval_model(model, train_data, metric)
    # evaluate the network on validation data
    val_loss_avg, val_r_sq = eval_model(model, val_data, metric)
    # return values
    return train_loss_avg, train_r_sq, val_loss_avg, val_r_sq


file_info = str(NUM_DATA_POINTS) + '_' + str(NUM_EPOCHS) + '_' + BAND + '_' + str(METRIC)
def save(filename):
    """
    Saves the saved model
    """
    save_filename = 'saved/' + filename + '_Reg.pt'
    torch.save(perf_model.state_dict(), save_filename)
    print('Saved as %s' % save_filename)

# define time logging method
def time_since(since):
    """
    Returns the time elapsed between now and 'since'
    """
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def adjust_learning_rate(optimizer, epoch, adjust_every):
    """
    Adjusts the learning rate of the optimizer based on the epoch
    Args:
       optimizer:      object, of torch.optim class 
       epoch:          int, epoch number
       adjust_every:   int, number of epochs after which adjustment is to done
    """
    if epoch > 1:
        if epoch % adjust_every == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5


# configure tensor-board logger
configure('runs/' + file_info + '_Reg' , flush_secs = 2)

## define training parameters
PRINT_EVERY = 1
ADJUST_EVERY = 1000
START = time.time()

try:
    print("Training for %d epochs..." % NUM_EPOCHS)
    for epoch in range(1, NUM_EPOCHS + 1):
        # perform training and validation
        train_loss, train_r_sq, val_loss, val_r_sq = train_and_validate(perf_model, criterion, perf_optimizer, training_data, validation_data, METRIC)
        # adjut learning rate
        adjust_learning_rate(perf_optimizer, epoch, ADJUST_EVERY)
        # log data for visualization later
        log_value('train_loss', train_loss, epoch)
        log_value('val_loss', val_loss, epoch)
        log_value('train_r_sq', train_r_sq, epoch)
        log_value('val_r_sq', val_r_sq, epoch)
        # print loss
        if epoch % PRINT_EVERY == 0:
            print('[%s (%d %.1f%%)]' % (time_since(START), epoch, float(epoch) / NUM_EPOCHS * 100))
            print('[%s %0.5f, %s %0.5f]'% ('Train Loss: ', train_loss, ' R-sq: ', train_r_sq))
            print('[%s %0.5f, %s %0.5f]'% ('Valid Loss: ', val_loss, ' R-sq: ', val_r_sq))

    print("Saving...")
    save(file_info)
except KeyboardInterrupt:
    print("Saving before quit...")
    save(file_info)

# test on testing data 
test_loss, test_r_sq = eval_model(perf_model, testing_data, METRIC)
print('[%s %0.5f, %s %0.5f]'% ('Testing Loss: ', test_loss, ' R-sq: ', test_r_sq))
