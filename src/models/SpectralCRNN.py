import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralCRNN(nn.Module):
    def __init__(self):
        super(SpectralCRNN, self).__init__()
        self.conv = nn.Sequential(
            # Conv Layer 1
            nn.Conv2d(1, 16, kernel_size=(3, 7), padding=(1,3)),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.MaxPool2d((2,4)),
            # Conv Layer 2
            nn.Conv2d(16, 32, kernel_size=(3, 7), padding=(1,3)),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d((3,5)),
            # Conv Layer 3
            nn.Conv2d(32, 64, kernel_size=(3, 7), padding=(1,3)),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d((3,5))
        )
        self.rnn = nn.GRU(320, 200, batch_first = True)
        self.fc = nn.Linear(100, 11)
        self.init_hidden(32)
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0), -1, out.size(3))
        out, _ = self.rnn(out, hidden)
        out = out[:,-1,:]
        out = self.fc(out)
        return f.LogSoftmax(out)
    def init_hidden(self, mini_batch_size):
        """
        Initializes the hidden state of the PitchContourAssessor module
        Args:
                mini_batch_size:    number of data samples in the mini-batch
        """
        self.hidden = Variable(torch.zeros(1, mini_batch_size, 200))
        if torch.cuda.is_available():
            self.hidden = self.hidden.cuda()