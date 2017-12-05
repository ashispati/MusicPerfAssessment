import torch
import time
import numpy as np
from torch import nn
from torch.autograd import Variable
from models.SpectralCRNN import SpectralCRNN_Reg_Dropout as SpectralCRNN
from tensorboard_logger import configure, log_value
from dataLoaders.SpectralDataset import SpectralDataset, SpectralDataLoader
from sklearn import metrics
from torch.optim import lr_scheduler

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def evaluate_classification(targets, predictions):
    r2 = metrics.r2_score(targets, predictions)
    targets = np.round(targets*10).astype(int)
    predictions = predictions * 10
    predictions[predictions < 0] = 0
    predictions[predictions > 10] = 10
    predictions = np.round(predictions).astype(int)
    accuracy = metrics.accuracy_score(targets, predictions)
    return r2, accuracy

# Configure tensorboard logger
configure('runs/MelSpec_reg_lr0.0001_big_ELU_Adam_noteacc' , flush_secs = 2)

# Parameteres for Spectral Representation
rep_params = {'method':'Mel Spectrogram', 'n_fft':2048, 'n_mels': 96, 'hop_length': 1024, 'normalize': True}

# Load Datasets
train_dataset = SpectralDataset('./dat/middle_2_data_3_train.dill', 1, rep_params)
train_dataloader = SpectralDataLoader(train_dataset, batch_size = 10, num_workers = 4, shuffle = True)

test_dataset = SpectralDataset('./dat/middle_2_data_3_test.dill', 1, rep_params)
test_dataloader = SpectralDataLoader(test_dataset, batch_size = 10, num_workers = 1, shuffle = True)

valid_dataset = SpectralDataset('./dat/middle_2_data_3_valid.dill', 1, rep_params)
valid_dataloader = SpectralDataLoader(valid_dataset, batch_size = 10, num_workers = 4, shuffle = True)

# Define Model
model = SpectralCRNN().cuda()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30], gamma = 0.1)

batch_time = AverageMeter()
data_time = AverageMeter()

train_loss = 0
validation_loss = 0

num_epochs = 250
best_val = 0.0
epoch_time = time.time()
for epoch in range(num_epochs):
    model.train()
    # scheduler.step()
    avg_loss = 0.0
    end = time.time()
    all_predictions = []
    all_targets = []
    losses = AverageMeter()
    for i, (data) in enumerate(train_dataloader):
        inputs, targets = data
        data_time.update(time.time() - end)
        inputs = Variable(inputs.cuda(), requires_grad = False)
        targets = Variable(targets.cuda(), requires_grad = False)
        targets = targets.view(-1,1)
        model.init_hidden(inputs.size(0))
        out = model(inputs)
        all_predictions.extend(out.data.cpu().numpy())
        all_targets.extend(targets.data.cpu().numpy())
        loss = criterion(out, targets)
        loss_value = loss.data[0]
        losses.update(loss_value, inputs.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
            (epoch + 1), (i + 1), len(train_dataloader), batch_time=batch_time,
            data_time=data_time, loss=losses))
    print('Epoch Completed. Validating')
    train_loss = losses.avg
    train_r2, train_accuracy = evaluate_classification(np.array(all_targets), np.array(all_predictions))
    
    model.eval()
    losses = AverageMeter()
    all_predictions = []
    all_targets = []
    for i, (data) in enumerate(valid_dataloader):
        inputs, targets = data
        data_time.update(time.time() - end)
        inputs = Variable(inputs.cuda(), requires_grad = False)
        targets = Variable(targets.cuda(), requires_grad = False)
        targets = targets.view(-1,1)
        model.init_hidden(inputs.size(0))
        out = model(inputs)
        all_predictions.extend(out.data.cpu().numpy())
        all_targets.extend(targets.data.cpu().numpy())
        loss = criterion(out, targets)
        loss_value = loss.data[0]
        losses.update(loss_value, inputs.size(0))
    print('Validating Completed. Loss: {}'.format(losses.avg))
    valid_loss = losses.avg
    val_r2, val_accuracy = evaluate_classification(np.array(all_targets), np.array(all_predictions))
    log_value('Train Loss', train_loss, epoch)
    log_value('Validation Loss', valid_loss, epoch)
    log_value('Training Accuracy', train_accuracy, epoch)
    log_value('Validation Accuracy', val_accuracy, epoch)
    log_value('Training R2', train_r2, epoch)
    log_value('Validation R2', val_r2, epoch)
    if val_r2 > best_val:
        best_val = val_r2
        torch.save(model, 'model_SpectralCRNN_reg_lr0.0001_big_ELU_Adam_noteacc')

