import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PitchContourAssessorLstmOnly(nn.Module):
    """
    Class to implement a deep neural model for music performance assessment using
	 pitch contours as input
    """

    def __init__(self):
        """
        Initializes the class with internal parameters for the different layers
        """
        super(PitchContourAssessorLstmOnly, self).__init__()
        # initialize interal parameters
        self.n_features = 64
        self.hidden_size = 128
        self.n_layers = 3
        # define the linear layer
        self.lin = nn.Linear(1, self.n_features)
        # define the LSTM module
        self.lstm = nn.LSTM(self.n_features, self.hidden_size, self.n_layers, batch_first=True)
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

        # compute the output of the linear layer
        lin_out = F.relu(self.lin(input.view(mini_batch_size, zero_pad_len, 1)))
   
        # compute the output of the lstm layer
        # transpose to ensure sequence length is dim 1 now
        lstm_out, self.hidden = self.lstm(lin_out)

        # extract final output of the lstm layer
		# TODO: take individual lengths into account
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
