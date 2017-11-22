import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PCConvNet(nn.Module):
    """
    Class to implement a deep neural model for music performance assessment using
	 pitch contours as input
    """

    def __init__(self):
        """
        Initializes the class with internal parameters for the different layers
        """
        super(PCConvNet, self).__init__()
        # initialize interal parameters
        self.kernel_size = 5
        self.stride = 3
        self.n0_features = 4
        self.n1_features = 8
        self.n2_features = 16
         # define the different convolutional modules
        self.conv0 = nn.Conv1d(1, self.n0_features, 7, 3) # output is (1000 - 7)/3 + 1 = 332
        self.conv0_bn = nn.BatchNorm1d(self.n0_features)
        self.conv1 = nn.Conv1d(self.n0_features, self.n1_features, self.kernel_size, self.stride) # output is (332 - 7)/3 + 1 = 76
        self.conv1_bn = nn.BatchNorm1d(self.n1_features) 
        self.conv2 = nn.Conv1d(self.n1_features, self.n2_features, self.kernel_size, self.stride) # output is (76 - 7)/3 + 1 = 24
        self.conv2_bn = nn.BatchNorm1d(self.n2_features) 
        self.conv3 = nn.Conv1d(self.n2_features, 1, 24, 1)


    def forward(self, input):
        """
        Defines the forward pass of the PitchContourAssessor module
        Args:
                input: 	torch Variable (mini_batch_size x zero_pad_len), of input pitch contours
                		mini_batch_size: 	size of the mini batch during one training iteration
            			zero_pad_len: 		length to which each input sequence is zero-padded
                		seq_lengths:		torch tensor (mini_batch_size x 1), length of each pitch contour
        """
         # get mini batch size from input
        mini_batch_size, sig_size = input.size()

        # compute the output of the convolutional layers
        conv0_out = F.relu(self.conv0_bn(self.conv0(input.view(mini_batch_size, 1, sig_size))))
        conv1_out = F.relu(self.conv1_bn(self.conv1(conv0_out)))
        conv2_out = F.relu(self.conv2_bn(self.conv2(conv1_out)))
        conv3_out = self.conv3(conv2_out)

        # compute final output
        final_output = torch.mean(conv3_out, 2)

        # return output
        return final_output
