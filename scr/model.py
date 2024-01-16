import torch
import torch.nn as nn
import torch.nn.functional as F  # Add this line


class SimpleLSTM(nn.Module):
    def __init__(self, input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers,
                 missed_char_dim):

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


class HangmanNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim,
                 output_dim, num_layers,
                 missed_char_dim, num_embeddings,
                 num_additional_features, dropout_rate=0.5):

        super(HangmanNN, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings, embedding_dim)  # Embedding layer

        # Dropout layer after embedding
        self.embedding_dropout = nn.Dropout(dropout_rate)

        # The input to the LSTM is the embedding_dim plus the number of additional features
        self.lstm_input_dim = embedding_dim + num_additional_features

        # Define the LSTM layer as bidirectional with dropout (except the last layer)
        self.lstm = nn.LSTM(self.lstm_input_dim, hidden_dim, num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout_rate if num_layers > 1 else 0)

        # Linear layer to process missed characters
        self.miss_linear = nn.Linear(missed_char_dim, hidden_dim)

        # Attention layer
        self.attention = nn.Linear(hidden_dim * 2, 1)

        # Dropout before the final linear layer
        self.before_linear_dropout = nn.Dropout(dropout_rate)

        # Adjust the input dimension of the linear layer
        self.linear = nn.Linear(
            hidden_dim * 4 + hidden_dim + num_additional_features, output_dim)

    def forward(self, features, original_seq_lens, missed_chars):
        print(f"Original features shape: {features.shape}")  # Debug
        print(f"Orginal seq len shape: {original_seq_lens.shape}")
        print(f"Missing character shape: {missed_chars.shape}")

        # Split features into encoded characters and additional features
        encoded_chars = features[:, :, 0].long()
        additional_features = features[:, :, 1:]
        print(f"Encoded chars shape: {encoded_chars.shape}")  # Debug
        # Debug
        print(f"Additional features shape: {additional_features.shape}")

        # Embedding for characters with dropout
        embedded_chars = self.embedding_dropout(self.embedding(encoded_chars))
        print(f"Embedded chars shape: {embedded_chars.shape}")  # Debug

        # Combine embedded characters with additional features
        lstm_input = torch.cat((embedded_chars, additional_features), dim=2)
        print(f"LSTM input shape: {lstm_input.shape}")  # Debug

        # Packing the input sequence
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            lstm_input, original_seq_lens.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_input)

        # Unpacking the sequence
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True)
        print(f"LSTM output shape: {output.shape}")  # Debug

        # Attention
        attention_weights = F.softmax(self.attention(output), dim=1)
        output = output * attention_weights
        print(f"Attention output shape: {output.shape}")  # Debug

        # Process missed characters
        missed_chars_processed = self.miss_linear(missed_chars)
        # Debug
        print(f"Missed chars processed shape: {missed_chars_processed.shape}")

        # Combine forward and backward hidden states
        hidden_forward = hidden[-2, :, :]
        hidden_backward = hidden[-1, :, :]
        print(f"Hidden forward shape: {hidden_forward.shape}")  # Debug
        print(f"Hidden backward shape: {hidden_backward.shape}")  # Debug

        # Concatenate LSTM output, processed missed characters, hidden states, and additional features
        concatenated = torch.cat((output, missed_chars_processed,
                                  hidden_forward, hidden_backward, additional_features), dim=2)
        print(f"Concatenated shape: {concatenated.shape}")  # Debug

        # Apply dropout before the final linear layer
        concatenated = self.before_linear_dropout(concatenated)

        # Apply the linear layer to the concatenated output
        out = self.linear(concatenated)
        print(f"Final output shape: {out.shape}")  # Debug

        return out
