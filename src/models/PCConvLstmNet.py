import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PCConvLstmNet(nn.Module):
    """
    Class to implement a deep neural model for music performance assessment using
	 pitch contours as input
    """

    def __init__(self):
        """
        Initializes the PitchContourAssessor class with internal parameters for the different layers
        """
        super(PCConvLstmNet, self).__init__()
        # initialize interal parameters
        self.kernel_size = 9
        self.stride = 5
        self.hidden_size = 32
        self.n_layers = 1
        self.n0_features = 6
        self.n1_features = 16
        #self.n2_features = 32
        #self.n3_features = 32
        #self.n4_features = 64
        #self.n5_features = 128
        # define the different convolutional modules
        self.conv0 = nn.Conv1d(1, self.n0_features, self.kernel_size, self.stride)
        #self.conv0_bn = nn.BatchNorm1d(self.n0_features)
        self.conv1 = nn.Conv1d(self.n0_features, self.n1_features, self.kernel_size, self.stride)
        #self.conv1_bn = nn.BatchNorm1d(self.n1_features)
        #self.conv2 = nn.Conv1d(self.n1_features, self.n2_features, self.kernel_size, self.stride)
        #self.conv2_bn = nn.BatchNorm1d(self.n2_features)
        #self.conv3 = nn.Conv1d(self.n2_features, self.n3_features, self.kernel_size, self.stride)
        #self.conv3_bn = nn.BatchNorm1d(self.n3_features)
        #self.conv4 = nn.Conv1d(self.n3_features, self.n4_features, self.kernel_size, self.stride)
        #self.conv4_bn = nn.BatchNorm1d(self.n4_features)
        #self.conv5 = nn.Conv1d(self.n4_features, self.n5_features, self.kernel_size, self.stride)
        # define the LSTM module
        self.lstm = nn.LSTM(self.n1_features, self.hidden_size,
                            self.n_layers, batch_first=True)
        # intialize the hidden state
        self.init_hidden(1)
        # define the final linear layer
        self.linear = nn.Linear(self.hidden_size, 1)

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
        mini_batch_size, zero_pad_len = input.size()

        # compute the output of the convolutional layer
        #conv0_out = F.relu(self.conv0_bn(self.conv0(input.view(mini_batch_size, 1, zero_pad_len))))
        #conv1_out = F.relu(self.conv1_bn(self.conv1(conv0_out)))
        #conv2_out = F.relu(self.conv2_bn(self.conv2(conv1_out)))
        #conv3_out = F.relu(self.conv3_bn(self.conv3(conv2_out)))
        #conv4_out = F.relu(self.conv4_bn(self.conv4(conv3_out)))
        conv0_out = F.max_pool1d(F.relu(self.conv0(input.view(mini_batch_size, 1, zero_pad_len))), 2)
        conv1_out = F.max_pool1d(F.relu(self.conv1(conv0_out)), 2)
        #conv2_out = F.max_pool1d(F.relu(self.conv2(conv1_out)), 2)
        #conv3_out = F.max_pool1d(F.relu(self.conv3(conv2_out)), 2)
        #conv4_out = F.max_pool1d(F.relu(self.conv4(conv3_out)), 2)
        #conv5_out = F.relu(self.conv5(conv4_out))
        # compute the output of the lstm layer
        # transpose to ensure sequence length is dim 1 now
        lstm_out, self.hidden = self.lstm(conv1_out.transpose(1, 2))

        # extract final output of the lstm layer
		# TODO: take into individual lengths
        mini_batch_size, lstm_seq_len, num_features = lstm_out.size()
        #print(lstm_seq_len)
        final_lstm_out = lstm_out[:, lstm_seq_len - 1, :]
        # compute output of the linear layer
        final_output = self.linear(final_lstm_out)

        # return output
        return final_output

    def init_hidden(self, mini_batch_size):
        """
        Initializes the hidden state of the PitchContourAssessor module
        Args:
                mini_batch_size: 	number of data samples in the mini-batch
        """
        self.hidden = Variable(torch.zeros(self.n_layers, mini_batch_size, self.hidden_size))
        if torch.cuda.is_available():
            self.hidden = self.hidden.cuda()
