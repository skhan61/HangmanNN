import torch
import torch.nn as nn
import torch.nn.functional as F  # Add this line


class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, missed_char_dim=27):
        super(SimpleLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # Define the LSTM layer as bidirectional
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True)

        # Linear layer to process missed characters
        self.miss_linear = nn.Linear(missed_char_dim, hidden_dim)

        # Adjust the input dimension of the linear layer
        self.linear = nn.Linear(hidden_dim * 2 + hidden_dim, output_dim)

    def forward(self, fets, original_seq_lens, missed_chars):
        # Packing the input sequence
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            fets, original_seq_lens.cpu(), batch_first=True, enforce_sorted=False)

        packed_output, (hidden, cell) = self.lstm(packed_input)

        # Unpacking the sequence
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True)

        # Process missed characters
        missed_chars_processed = self.miss_linear(missed_chars)

        # Concatenate LSTM output and processed missed characters
        concatenated = torch.cat((output, missed_chars_processed), dim=2)

        # Apply the linear layer to the concatenated output
        out = self.linear(concatenated)

        # out = torch.sigmoid(out)

        return out
