import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PCConvNet(nn.Module):
    """
    Class to implement a deep neural model for music performance assessment using
	 pitch contours as input
    """

    def __init__(self, mode):
        """
        Initializes the class with internal parameters for the different layers
        Args:
            mode: 0,1 specifying different minimum input size, 0: 1000, 1:500
        """
        super(PCConvNet, self).__init__()
        if mode == 0: # for minimum input size of 1000
            # initialize model internal parameters
            self.kernel_size = 7
            self.stride = 3
            self.n0_features = 4
            self.n1_features = 8
            self.n2_features = 16
            # define the different convolutional modules
            self.conv0 = nn.Conv1d(1, self.n0_features, self.kernel_size, self.stride) # output is (1000 - 7)/3 + 1 = 332
            self.conv0_bn = nn.BatchNorm1d(self.n0_features)
            #self.maxpool0 = nn.MaxPool1d(3) # output is 996 / 3 = 332
            self.conv1 = nn.Conv1d(self.n0_features, self.n1_features, self.kernel_size, self.stride) # output is (332 - 7)/3 + 1 = 109
            self.conv1_bn = nn.BatchNorm1d(self.n1_features)
            #self.maxpool1 = nn.MaxPool1d(3) # output is 326 / 3 = 108
            self.conv2 = nn.Conv1d(self.n1_features, self.n2_features, self.kernel_size, self.stride) # output is (109 - 7)/3 + 1 = 35
            self.conv2_bn = nn.BatchNorm1d(self.n2_features)
            #self.maxpool2 = nn.MaxPool1d(3) # output if 102 / 3 = 34
            self.conv3 = nn.Conv1d(self.n2_features, 1, 35, 1)
            self.conv3_bn = nn.BatchNorm1d(1)
        elif mode == 1: # for minimum input size of 500
            # initialize model internal parameters
            self.kernel_size = 5
            self.stride = 2
            self.n0_features = 4
            self.n1_features = 8
            self.n2_features = 16
            # define the convolutional modelues
            self.conv0 = nn.Conv1d(1, self.n0_features, self.kernel_size, self.stride) # output is (500 - 5)/2 + 1 = 248
            self.conv0_bn = nn.BatchNorm1d(self.n0_features)
            #self.maxpool0 = nn.MaxPool1d(2) # output is 496 / 2 = 248
            self.conv1 = nn.Conv1d(self.n0_features, self.n1_features, self.kernel_size, self.stride) # output is (248 - 5)/2 + 1 = 122
            self.conv1_bn = nn.BatchNorm1d(self.n1_features)
            #self.maxpool1 = nn.MaxPool1d(2) # output is 244 / 2 = 122
            self.conv2 = nn.Conv1d(self.n1_features, self.n2_features, 7, 4) # output is (122 - 7)/4 + 1 = 29
            self.conv2_bn = nn.BatchNorm1d(self.n2_features)
            #self.maxpool2 = nn.MaxPool1d(4) # output is 116 / 4 = 29
            self.conv3 = nn.Conv1d(self.n2_features, 1, 28, 1)
            self.conv3_bn = nn.BatchNorm1d(1)

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
        #print(input.size())
        # compute the output of the convolutional layers
        conv0_out = F.leaky_relu(self.conv0_bn(self.conv0(input.view(mini_batch_size, 1, sig_size))))
        #print(conv0_out.size())
        #conv0_out = self.maxpool0(conv0_out)
        #print(conv0_out.size())
        conv1_out = F.leaky_relu(self.conv1_bn(self.conv1(conv0_out)))
        #print(conv1_out.size())
        #conv1_out = self.maxpool1(conv1_out)
        #print(conv1_out.size())
        conv2_out = F.leaky_relu(self.conv2_bn(self.conv2(conv1_out)))
        #print(conv2_out.size())
        #conv2_out = self.maxpool2(conv2_out)
        #print(conv2_out.size())
        conv3_out = F.relu(self.conv3_bn(self.conv3(conv2_out)))
        #print(conv3_out.size())

        # compute final output
        final_output = torch.mean(conv3_out, 2)
        #print(final_output.size())
        # return output
        return final_output
