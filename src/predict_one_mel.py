import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from models.SpectralCRNN import SpectralCRNN_Reg_Dropout as SpectralCRNN
#from models.SpectralCRNN import SpectralCRNN_Reg_big as SpectralCRNN
from dataLoaders.SpectralDataset import SpectralDataset, SpectralDataLoader
from sklearn import metrics

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

# Set constants and other important values
BAND = 'symphonic'
NAME = 'MelSpec_reg_lr0.0001_dropout_Adam_tone_middle_triangularNoise_test'
criterion = nn.MSELoss()

def predict_one_mel(BAND, filename, inputrep):
    NAME = filename
    # Parameteres for Spectral Representation
    rep_params = {'method':inputrep, 'n_fft':2048, 'n_mels': 96, 'hop_length': 1024, 'normalize': True}

    test_dataset = SpectralDataset('/home/mfarren/dat/test_' + BAND + '.dill', 3, rep_params)
    test_dataloader = SpectralDataLoader(test_dataset, batch_size = 10, num_workers = 1, shuffle = True)

    test_loss = 0

    model = SpectralCRNN().cuda()

    #model.load_state_dict(torch.load('model_' + NAME))
    model = torch.load('saved_runs/model_' + NAME)

    model.eval()
    losses = AverageMeter()
    all_predictions = []
    all_targets = []
    print("Evaluating...")
    for i, (data) in enumerate(test_dataloader):
        inputs, targets = data
        inputs = Variable(inputs.cuda(), requires_grad = False)
        targets = Variable(targets.cuda(), requires_grad = False)
        targets = targets.view(-1,1)
        model.init_hidden(inputs.size(0))
        out = model(inputs)
        all_predictions.extend(out.data.cpu().numpy())
        all_targets.extend(targets.data.cpu().numpy())
        loss = criterion(out, targets)
        loss_value = loss.data.item()
        losses.update(loss_value, inputs.size(0))
    print('Testing Completed. Loss: {}'.format(losses.avg))
    test_loss = losses.avg
    test_r2, test_accuracy = evaluate_classification(np.array(all_targets), np.array(all_predictions))
    print('Test r^2: ', str(test_r2))
    print('Test loss: ', str(test_loss))

    """
    vars = [test_loss, test_r2]
    with open('test_results_mel.txt', 'w') as file:
        for i in range(len(vars)):
            if i == 0:
                file.write("Test loss:")
                file.write(str(vars[i]))
            elif i == 1:
                file.write("r-squared:\n")
                file.write(str(vars[i]) + '\n')
            else:
                file.write('\n')
                file.write(str(vars[i]) + '\n')
    """

#predict_one_mel('symphonic', 'ACF_reg_lr0.0001_dropout_Adam_musicality_symphonic', 'ACF')
