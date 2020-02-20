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
from models.PCConvNet import PCConvNet
from models.PCConvLstmNet import PCConvLstmNet
from dataLoaders.PitchContourDataset import PitchContourDataset
from dataLoaders.PitchContourDataloader import PitchContourDataloader
from sklearn import metrics
import eval_utils
from vis_utils import visualize_grid, vis_layer

np.set_printoptions(precision=4)

# check is cuda is available and print result
CUDA_AVAILABLE = torch.cuda.is_available()
print('Running on GPU: ', CUDA_AVAILABLE)
if CUDA_AVAILABLE != True:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

# initializa training parameters
RUN = 13
NUM_EPOCHS = 2000
NUM_DATA_POINTS = 1550
NUM_BATCHES = 10
BAND = 'symphonic'
SEGMENT = '2'
METRIC = 0 # 0: Musicality, 1: Note Accuracy, 2: Rhythmic Accuracy, 3: Tone Quality
MTYPE = 'conv'
CTYPE = 0

# define model
if MTYPE == 'conv':
    perf_model = PCConvNet(0)
elif MTYPE == 'lstm':
    perf_model = PCConvLstmNet()
if CUDA_AVAILABLE:
    perf_model.cuda()
criterion = nn.MSELoss() 

# read the model
filename = str(NUM_DATA_POINTS) + '_' + str(NUM_EPOCHS) + '_' + BAND + '_' + str(METRIC) + '_' + str(RUN) + '_' + MTYPE + '_best_Reg'
if torch.cuda.is_available():
    perf_model.cuda()
    perf_model.load_state_dict(torch.load('saved/' + filename + '.pt'))
else:
    perf_model.load_state_dict(torch.load('saved/' + filename + '.pt', map_location=lambda storage, loc: storage))
'''
for name, param in perf_model.named_parameters():
    #if name == 'conv.0.weight':
    if param.requires_grad:
        
        if name == 'conv.3.weight':
            nparray = param.data.numpy()
            nparray = np.reshape(nparray, (32,7))
            grid  = vis_layer(nparray)
            plt.imshow(grid.astype('uint8'))
            plt.savefig('/Users/Som/Desktop/DLFBAFigures/' + BAND + '_' + str(METRIC) + '_2ndLayer.png', format='png', dpi=1000)
            #plt.show()
        
        if name == 'conv.0.weight':
            layer1w = param.data.clone()
            nparray = param.data.numpy()
            nparray = np.reshape(nparray, (4,7))
            grid  = vis_layer(nparray)
            plt.imshow(grid.astype('uint8'))
            ylabels = [1, 2, 3, 4]
            plt.yticks([0, 1, 2, 3], ylabels)
            plt.ylabel('Channel Number')
            plt.xlabel('Kernel Width')
            plt.savefig('/Users/Som/Desktop/DLFBAFigures/' + BAND + '_' + str(METRIC) + '_1stLayer.png', format='png', dpi=300)
            plt.show()
        if name == 'conv.0.bias':
            layer1b = param.data.clone()
        #nparray = param.data.numpy()
        #print(name, param.data)
        #grid  = vis_layer(np.reshape(nparray, (4, 7)))
        #plt.imshow(grid.astype('uint8'))
        #plt.savefig('/Users/Som/Desktop/DLFBAFigures/' + BAND + '_' + str(METRIC) + '_1stLayer.png', format='png', dpi=1000)
        #plt.show()

layer1w = np.reshape(layer1w.numpy(), (28))
for p in layer1w: print(p)
#print(layer1b)
'''
# initialize dataset, dataloader and created batched data
file_name = BAND + '_' + str(SEGMENT) + '_data'
if sys.version_info[0] < 3:
    data_path = 'dat/' + file_name + '.dill'
else:
    data_path = 'dat/' + file_name + '_3.dill'
dataset = PitchContourDataset(data_path)
dataloader = PitchContourDataloader(dataset, NUM_DATA_POINTS, NUM_BATCHES)
_, _, vef, _, tef, _ = dataloader.create_split_data(1000, 500)
# test on full length data
#val_loss, val_r_sq, val_accu, val_accu2, pred, target = eval_utils.eval_model(perf_model, criterion, vef, METRIC, MTYPE, CTYPE, 1)
#print('[%s %0.5f, %s %0.5f, %s %0.5f %0.5f]'% ('Valid Loss: ', val_loss, ' R-sq: ', val_r_sq, ' Accu:', val_accu, val_accu2))
test_loss, test_r_sq, test_accu, test_accu2, pred, target = eval_utils.eval_model(perf_model, criterion, tef, METRIC, MTYPE, CTYPE, 1)
print('[%s %0.5f, %s %0.5f, %s %0.5f %0.5f]'% ('Testing Loss: ', test_loss, ' R-sq: ', test_r_sq, ' Accu:', test_accu, test_accu2))

# convert to numpy
if torch.cuda.is_available():
    pred = pred.clone().cpu().numpy()
    target = target.clone().cpu().numpy()
else:
    pred = pred.clone().numpy()
    target = target.clone().numpy()
#a = np.absolute((pred - target))

# compute correlation coefficient
R, p = ss.pearsonr(pred, target)
print(R, p)


#print('Target')
#for p in target: print(p)
#print('Pred')
#for p in pred: print(p)

# sort based ascending order of prediction residual
idx = 76
data_point = tef[idx]
print(target[idx], pred[idx])
X = data_point['pitch_tensor']
y = data_point['score_tensor'][:, METRIC]
smap = np.absolute(eval_utils.compute_saliency_maps(X, y, perf_model).view(-1).numpy())
thres =  3 * np.median(smap)
print(thres)

X_np = np.around(X.view(-1).numpy(), decimals=3)
X_map = X_np.copy()
X_np[X_np == 0.0] = float('nan')
X_map[smap < thres] = float('nan')
plt.plot(X_np, 'k', linewidth=6.0)
plt.plot(X_map, 'r.', markersize=3)

plt.ylabel('Normalized Pitch', fontsize=18)
plt.xlabel('Blocks',fontsize=18)
plt.tick_params(labelsize=12)
plt.savefig('/Users/Som/Desktop/DLFBAFigures/SaliencyMaps/' + BAND + '_' + str(METRIC) + '_' + str(idx) + '.png', format='png', dpi=300)
plt.show()




