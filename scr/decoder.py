import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers,
                 output_dim, missed_char_dim,
                 dropout_prob=0.5):

        # print(f"Model initilized...")

        super(SimpleLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.missed_char_dim = missed_char_dim

        # Define the LSTM layer as bidirectional
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout_prob if num_layers > 1 else 0.0)

        # Linear layer to process missed characters
        self.miss_linear = nn.Linear(missed_char_dim, hidden_dim)

        # Calculate the linear layer input dimension
        # 2*hidden_dim (LSTM) + 2*hidden_dim (hidden state) + hidden_dim (missed chars)
        linear_input_dim = hidden_dim * 5

        # Initialize the linear layer
        self.linear = nn.Linear(linear_input_dim, output_dim)

    def forward(self, fets, original_seq_lens, missed_chars):
        # Debugging print
        # print("Input features shape:", fets.shape)
        # print("Original Seq shape: ", original_seq_lens.shape)
        # print("Missed Chars shape: ", missed_chars.shape)

        # Packing the input sequence
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            fets, original_seq_lens.cpu(), batch_first=True, enforce_sorted=False)
        # print("Packed input shape:", packed_input.data.shape)

        # LSTM processing
        packed_output, (hidden, cell) = self.lstm(packed_input)
        # print("Packed output shape:", packed_output.data.shape)
        # print("Hidden shape: ", hidden.shape)

        # Unpacking the sequence
        unpacked_output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True)
        # print("Unpacked output shape:", unpacked_output.shape)

        # Process final hidden state
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_dim)
        hidden = hidden[-1]
        hidden = hidden.permute(1, 0, 2)
        hidden = hidden.contiguous().view(hidden.shape[0], -1)
        # print()
        # print(
        #    f"Misse character process shape after unpadding: {missed_chars_processed.shape}")

        # print(f"unpacked_output size: {unpacked_output.size(-1)}")
        # print(f"hidden size: {hidden.size(-1)}")
        # print(
        #    f"missed_chars_processed size: {missed_chars_processed.size(-1)}")

        # Process missed characters
        missed_chars_processed = self.miss_linear(missed_chars)
        missed_chars_processed = missed_chars_processed[:, :unpacked_output.size(
            1), :]

        # Combine all elements
        combined = torch.cat(
            [unpacked_output, hidden.unsqueeze(1).repeat(1,
                                                         unpacked_output.size(1), 1),
                missed_chars_processed], dim=2)

        # print(f"Combine shape: {combined.shape}")

        # Apply the linear layer to the concatenated output
        out = self.linear(combined)

        return out


def test_simple_lstm():
    # Test parameters
    batch_size = 10
    max_seq_length = 10
    max_word_length = 29
    num_embeddings = 28
    embedding_dim = 50
    num_features = 5
    # Calculate the input dimension for the LSTM (SimpleLSTM decoder)
    input_dim = max_word_length * embedding_dim \
        + (num_features - 1) * max_word_length
    hidden_dim = 256
    output_dim = 28
    num_layers = 2
    missed_char_dim = 28

    # Instantiate the SimpleLSTM model
    model = SimpleLSTM(input_dim=input_dim,
                       hidden_dim=hidden_dim,
                       output_dim=output_dim,
                       num_layers=num_layers,
                       missed_char_dim=missed_char_dim)

    # Generate dummy input data
    features = torch.rand(batch_size, max_seq_length, input_dim)
    original_seq_lens = torch.randint(
        1, max_seq_length + 1, (batch_size,)).sort(descending=True)[0]
    missed_chars = torch.rand(batch_size, max_seq_length, missed_char_dim)

    # Test the SimpleLSTM with the dummy data
    output = model(features, original_seq_lens, missed_chars)

    print()

    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    test_simple_lstm()
