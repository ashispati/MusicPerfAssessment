from __future__ import print_function
import argparse
import gc
import math
import numpy as np
import os
import os.path as op
import scipy.stats as ss
import sys
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from dataLoaders.RawAudioDataset import RawAudioDataset
from dataLoaders.RawAudioDataLoader import RawAudioDataLoader
from models.RawAudioNet import RawAudioNet
from sklearn import metrics
from tensorboard_logger import configure, log_value
from torch.autograd import Variable

# set manual random seed for reproducibility
torch.manual_seed(1)

# check is cuda is available and print result
CUDA_AVAILABLE = torch.cuda.is_available()
print('Running on GPU: ', CUDA_AVAILABLE)
if CUDA_AVAILABLE != True:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

# initializa training parameters
parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', default=100, type=int,
                    help="number of training epochs.")
parser.add_argument('-nd', '--data_points', default=1100, type=int,
                    help="number of data points.")
parser.add_argument('-nb', '--num_batches', default=110, type=int,
                    help="number of batches to use in training.")
parser.add_argument('-b', '--band', default="middle",
                    help="frequency (band) of data to use.")
parser.add_argument('-s', '--segment', default="2",
                    help="I have no idea what this is tbh.")
parser.add_argument('-lr', '--learning_rate', default=1e-2, type=float,
                    help="learning rate for sgd.")
parser.add_argument('-wd', '--weight_decay', default=1e-5, type=float,
                    help="weight decay parameter for sgd.")
parser.add_argument('-m', '--momentum', default=0.9, type=float,
                    help="learning rate for sgd.")
parser.add_argument('-mt', '--metric', default=0, type=int, choices=set((0, 1, 2, 3)),
                    help=("the metric by which to judge a data point. "
                          "0: musicality, 1: note accuracy, "
                          "2: rhythmic accuracy, 3: tone quality"))
args = parser.parse_args()

NUM_EPOCHS = args.epochs
NUM_DATA_POINTS = args.data_points
NUM_BATCHES = args.num_batches
BAND = args.band
SEGMENT = args.segment
L_RATE = args.learning_rate
W_DECAY = args.weight_decay
MOMENTUM = args.momentum
METRIC = args.metric # 0: Musicality, 1: Note Accuracy, 2: Rhythmic Accuracy, 3: Tone Quality

print('\nTraining Args: ')
print('NUM_EPOCHS: {}'.format(NUM_EPOCHS))
print('NUM_DATA_POINTS: {}'.format(NUM_DATA_POINTS))
print('NUM_BATCHES: {}'.format(NUM_BATCHES))
print('BAND: {}'.format(BAND))
print('SEGMENT: {}'.format(SEGMENT))
print('L_RATE: {}'.format(L_RATE))
print('W_DECAY: {}'.format(W_DECAY))
print('MOMENTUM: {}'.format(MOMENTUM))
print('METRIC: {}\n'.format(METRIC))

train_dataset = RawAudioDataset(op.join('dat', 'train.dill'))
train_dataloader = RawAudioDataLoader(train_dataset, NUM_DATA_POINTS, NUM_BATCHES)

valid_dataset = RawAudioDataset(op.join('dat', 'valid.dill'))
valid_dataloader = RawAudioDataLoader(valid_dataset, num_batches=int(len(valid_dataset)/10)) 

test_dataset = RawAudioDataset(op.join('dat', 'test.dill'))
test_dataloader = RawAudioDataLoader(test_dataset, num_batches=int(len(test_dataset)/10))

# tr1, v1, te1 = dataloader.create_split_data(1000, 500)
# training_data = tr1
# validation_data = v1
# testing_data = te1

# split batches into training, validation and testing
#training_data = batched_data[0:8]
#validation_data = batched_data[8:9]
#testing_data = batched_data[9:10]

# create batched data for all datasets
train_data = train_dataloader.create_batched_data()
valid_data = valid_dataloader.create_batched_data()
test_data = test_dataloader.create_batched_data()

## initialize model
perf_model = RawAudioNet()
if CUDA_AVAILABLE:
    perf_model.cuda()
criterion = nn.MSELoss()
# perf_optimizer = optim.SGD(perf_model.parameters(), lr=L_RATE, momentum=MOMENTUM, weight_decay=W_DECAY)
perf_optimizer = optim.Adam(perf_model.parameters(), lr=L_RATE)
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

    # compute r-sq score
    r_sq = metrics.r2_score(target_np, pred_np)
    #print(pred_np)
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
        model:          object, trained model of RawAudioAssessor class
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
        audio_tensor = data[batch_idx]['audio_tensor']
        ratings_tensor = data[batch_idx]['ratings_tensor'][:, metric]
        # prepare data for input to model
        model_input = audio_tensor.clone()
        model_target = ratings_tensor.clone()
        # convert to cuda tensors if cuda available
        if CUDA_AVAILABLE:
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
        loss = criterion(model_output, model_target)
        loss_avg += loss.data[0]
        # concatenate target and pred for computing validation metrics
        pred = torch.cat((pred, model_output.data.view(-1)), 0) if pred.size else model_output.data.view(-1)
        target = torch.cat((target, ratings_tensor), 0) if target.size else ratings_tensor
    r_sq, accu, accu2 = eval_regression(target, pred)
    loss_avg /= num_batches
    return loss_avg, r_sq, accu, accu2

def train(model, criterion, optimizer, data, metric):
    """
    Returns the model performance metrics
    Args:
        model:          object, trained model of RawAudioAssessor class
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
        optimizer.zero_grad()
        loss = 0

        # extract audio tensor and score for the batch
        audio_tensor = data[batch_idx]['audio_tensor']
        ratings_tensor = data[batch_idx]['ratings_tensor'][:, metric]

        # prepare data for input to model
        model_input = audio_tensor.clone()
        model_target = ratings_tensor.clone()
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
        # print('model_input: {}'.format(model_input))
        # print('model_output: {}'.format(model_output))

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
        model:          object, trained model of RawAudioAssessor class
        criterion:      object, of torch.nn.Functional class which defines the loss 
        optimizer:      object, of torch.optim class which defines the optimization algorithm
        train_data:     list, batched training data
        val_data:       list, batched validation data
        metric:         int, from 0 to 3, which metric to evaluate against
    """
    # train the network
    train(model, criterion, optimizer, train_data, metric)
    # evaluate the network on train data
    train_loss_avg, train_r_sq, train_accu, train_accu2 = eval_model(model, train_data, metric)
    # evaluate the network on validation data
    val_loss_avg, val_r_sq, val_accu, val_accu2 = eval_model(model, val_data, metric)
    # return values
    return train_loss_avg, train_r_sq, train_accu, train_accu2, val_loss_avg, val_r_sq, val_accu, val_accu2


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
BEST_VAL_R2 = -1

final_results = {}
try:
    print("Training for %d epochs..." % NUM_EPOCHS)
    for epoch in range(1, NUM_EPOCHS + 1):
        # perform training and validation
        (train_loss, train_r_sq, train_accu, train_accu2, val_loss,
         val_r_sq, val_accu, val_accu2) = train_and_validate(perf_model, criterion,
                                                             perf_optimizer, train_data,
                                                             valid_data, METRIC)
        BEST_VAL_R2 = max(val_r_sq, BEST_VAL_R2)
        # adjut learning rate
        adjust_learning_rate(perf_optimizer, epoch, ADJUST_EVERY)
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
            print('[time_elapsed: %s, epoch: %d, percent_complete:  %.1f%%]'%(time_since(START), epoch, float(epoch) / NUM_EPOCHS * 100))
            print('[%s %0.5f, %s %0.5f, %s %0.5f %0.5f]'% ('Train Loss: ', train_loss, ' R-sq: ', train_r_sq, ' Accu:', train_accu, train_accu2))
            print('[%s %0.5f, %s %0.5f, %s %0.5f %0.5f]'% ('Valid Loss: ', val_loss, ' R-sq: ', val_r_sq, ' Accu:', val_accu, val_accu2))
            print('[best validation r2 value seen: %0.5f]'%(BEST_VAL_R2))
    final_results.update({'final_train_loss': train_loss,
                          'final_val_loss': val_loss,
                          'final_train_r_sq': train_r_sq,
                          'final_val_r_sq': val_r_sq,
                          'final_train_accuracy': train_accu,
                          'final_val_accuracy': val_accu,
                          'final_train_accuracy_2': train_accu2,
                          'final_val_accuracy_2': val_accu2,
                          'best_val_r_sq': BEST_VAL_R2})
    print("Saving...")
    save(file_info)
except KeyboardInterrupt:
    print("Saving before quit...")
    save(file_info)

# test on testing data
test_loss, test_r_sq, test_accu, test_accu2 = eval_model(perf_model, test_data, METRIC)
print('[%s %0.5f, %s %0.5f, %s %0.5f %0.5f]'% ('Testing Loss: ', test_loss, ' R-sq: ', test_r_sq, ' Accu:', test_accu, test_accu2))
final_results.update({'test_loss': test_loss,
                      'test_r_sq': test_r_sq,
                      'test_accuracy': test_accu,
                      'test_accuracy_2': test_accu2})

print("Saving final results.")
datestring = datetime.utcnow().date().strptime('%Y%m%d')
with open('save/rawaudionet_final_results_metric_%d_%s.txt'%(METRIC, datestring), 'wb') as outfile:
    for k,v in final_results.iteritems():
        outfile.write('%s:\t%s\n'%(k, v))
    outfile.close()

# test of full length data
# test_loss, test_r_sq, test_accu, test_accu2 = eval_model(perf_model, test_data, METRIC)
# print('[%s %0.5f, %s %0.5f, %s %0.5f %0.5f]'% ('Testing Loss: ', test_loss, ' R-sq: ', test_r_sq, ' Accu:', test_accu, test_accu3))

