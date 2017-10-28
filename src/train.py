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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorboard_logger import configure, log_value

# set manual random seed for reproducibility
torch.manual_seed(1)

# check is cuda is available and print result
CUDA_AVAILABLE = torch.cuda.is_available()
print(CUDA_AVAILABLE)

# initializa training parameters
NUM_EPOCHS = 50
NUM_DATA_POINTS = 100
NUM_BATCHES = 10
BAND = 'middle'
INSTRUMENT = 'Alto Saxophone'
SEGMENT = '2'
CRITERIA = 0

# initialize dataset, dataloader and created batched data
file_name = BAND + '_' + INSTRUMENT[:4] + '_' + str(SEGMENT) + '_data'
if sys.version_info[0] < 3:
    data_path = 'dat/' + file_name + '.dill'
else:
    data_path = 'dat/' + file_name + '_3.dill'
dataset = PitchContourDataset(data_path)
dataloader = PitchContourDataloader(dataset, NUM_DATA_POINTS, NUM_BATCHES)
batched_data = dataloader.create_batched_data()
print(batched_data[0])

## initialize model
perf_model = PitchContourAssessor()
criterion = nn.MSELoss()
LR_RATE = 0.001
perf_optimizer = optim.SGD(perf_model.parameters(), LR_RATE)
print(perf_model)

# define training method
def train(batched_data):
    """
    Defines the training cycle for the input batched data
    """
    num_batches = len(batched_data)
    loss_avg = 0
	# iterate over batches
    for batch_idx in range(num_batches):
		# clear gradients and loss for both melody and rhythm networks
        perf_model.zero_grad()
        loss = 0 

        # extract pitch tensor and score for the batch
        pitch_tensor = batched_data[batch_idx]['pitch_tensor']
        score_tensor = batched_data[batch_idx]['score_tensor'][:, 0] # 0 for musicality

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
        model_output = perf_model(model_input)
        # compute loss
        loss = criterion(model_output, model_target)
        # compute backward pass and step    
        loss.backward()
        perf_optimizer.step()
        # add loss
        loss_avg += loss.data[0]

    return loss_avg / num_batches

def save():
    """
    Saves the saved model
    """
    file_info = str(NUM_DATA_POINTS) + '_' + str(NUM_EPOCHS)
    save_filename = 'saved/' + file_info + 'PerfModel.pt'
    torch.save(perf_model.state_dict(), save_filename)
    print('Saved as %s' % save_filename)
    sys.exit()

# define time logging method
def timeSince(since):
    """
    Returns the time elapsed between now and 'since'
    """
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

# configure tensor-board logger
configure('runs/' + str(NUM_DATA_POINTS) + '_' + str(NUM_EPOCHS), flush_secs = 2)

## define training parameters
print_every = 1
start = time.time()
all_losses = []

try:
    print("Training for %d epochs..." % NUM_EPOCHS)
    for epoch in range(1, NUM_EPOCHS + 1):
        # train the network
        loss = train (batched_data)

        # log data for visualization later
        log_value('loss', loss, epoch)

        # print loss
        if epoch % print_every == 0:
            print('[%s (%d %.1f%%) %.5f]' % (timeSince(start), epoch,
                    float(epoch) / NUM_EPOCHS * 100, loss))
    print("Saving...")
    save()
except KeyboardInterrupt:
    print("Saving before quit...")
    save()
