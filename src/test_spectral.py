import torch
import time
import numpy as np
from torch import nn
from scipy.stats import pearsonr
from torch.autograd import Variable
from models.SpectralCRNN import SpectralCRNN_Reg_Dropout as SpectralCRNN
from tensorboard_logger import configure, log_value
from dataLoaders.SpectralDataset import SpectralDataset, SpectralDataLoader
from sklearn import metrics
from torch.optim import lr_scheduler

def evaluate_classification(targets, predictions):
    r2 = metrics.r2_score(targets, predictions)
    corrcoef, p = pearsonr(targets, predictions)
    targets = np.round(targets*10).astype(int)
    predictions = predictions * 10
    predictions[predictions < 0] = 0
    predictions[predictions > 10] = 10
    predictions = np.round(predictions).astype(int)
    accuracy = metrics.accuracy_score(targets, predictions)
    return r2, accuracy, corrcoef[0], p[0]

def evaluate_model(model, dataloader):
    model.eval()
    all_predictions = []
    all_targets = []
    for i, (data) in enumerate(dataloader):
        inputs, targets = data
        inputs = Variable(inputs.cuda(), requires_grad = False)
        targets = Variable(targets.cuda(), requires_grad = False)
        targets = targets.view(-1,1)
        model.init_hidden(inputs.size(0))
        out = model(inputs)
        all_predictions.extend(out.data.cpu().numpy())
        all_targets.extend(targets.data.cpu().numpy())
    return evaluate_classification(np.array(all_targets), np.array(all_predictions))

rep_params = {'method':'Mel Spectrogram', 'n_fft':2048, 'n_mels': 96, 'hop_length': 1024, 'normalize': True}

train_dataset = SpectralDataset('./dat/middle_2_data_3_train.dill', 1, rep_params)
train_dataloader = SpectralDataLoader(train_dataset, batch_size = 10, num_workers = 4, shuffle = True)

test_dataset = SpectralDataset('./dat/middle_2_data_3_test.dill', 1, rep_params)
test_dataloader = SpectralDataLoader(test_dataset, batch_size = 10, num_workers = 1, shuffle = True)

valid_dataset = SpectralDataset('./dat/middle_2_data_3_valid.dill', 1, rep_params)
valid_dataloader = SpectralDataLoader(valid_dataset, batch_size = 10, num_workers = 4, shuffle = True)

model_path = 'model_SpectralCRNN_reg_lr0.0001_big_ELU_Adam_noteacc'
model = SpectralCRNN().cuda()
model = torch.load(model_path)

criterion = nn.MSELoss()

train_metrics = evaluate_model(model, train_dataloader)
val_metrics = evaluate_model(model, valid_dataloader)
test_metrics = evaluate_model(model, test_dataloader)

print("Training Rsq:")