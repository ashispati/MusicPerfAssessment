import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PitchContourAssessor(nn.Module):
    """
    Class to implement a deep neural model for music performance assessment using
	 pitch contours as input
    """

    def __init__(self):
        """
        Initializes the PitchContourAssessor class with internal parameters for the different layers
        """
        super(PitchContourAssessor, self).__init__()
        # initialize interal parameters
        self.kernel_size = 13
        self.stride = 6
        self.hidden_size = 128
        self.n_layers = 3
        self.n_features = 32
        # define the different convolutional modules
        self.conv1 = nn.Conv1d(1, 6, self.kernel_size, self.stride)
        self.conv2 = nn.Conv1d(6, 16, self.kernel_size, self.stride)
        self.conv3 = nn.Conv1d(16, self.n_features,
                               self.kernel_size, self.stride)

        # define the LSTM module
        self.lstm = nn.LSTM(self.n_features, self.hidden_size,
                            self.n_layers, batch_first=True)

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
        conv1_out = F.relu(self.conv1(input.view(mini_batch_size, 1, zero_pad_len)))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))

        # compute the output of the lstm layer
        self.init_hidden(mini_batch_size)
        # transpose to ensure sequence length is dim 1 now
        lstm_out, self.hidden = self.lstm(conv3_out.transpose(1, 2))

        # extract final output of the lstm layer
		# TODO: take into individual lengths
        mini_batch_size, lstm_seq_len, num_features = lstm_out.size()
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
