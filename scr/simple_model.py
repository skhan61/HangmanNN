import torch
import torch.nn as nn
import torch.nn.functional as F  # Add this line
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score

from scr.base_model import BaseModel


class SimpleLSTM(BaseModel):
    def __init__(self, config):
        super(SimpleLSTM, self).__init__(config)
        self.embedding_dim = config.get('embedding_dim', 200)
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_layers = config.get('num_layers', 2)
        # Defaulting to 26 letters + 1
        self.vocab_size = config.get('vocab_size', 27)
        self.max_word_length = config.get('max_word_length', 5)
        self.input_feature_size = config.get('input_feature_size', 5)
        self.use_embedding = config.get('use_embedding', True)
        self.lr = config.get('lr', 0.001)  # Default learning rate

        # Device handling
        self.device = torch.device("cuda" if torch.cuda.is_available(
        ) and config.get('use_cuda', False) else "cpu")

        if self.use_embedding:
            self.embedding = nn.Embedding(
                self.vocab_size + 1, self.embedding_dim)
            rnn_input_size = self.embedding_dim * self.max_word_length + \
                (self.input_feature_size - 1) * self.max_word_length
        else:
            rnn_input_size = self.input_feature_size * self.max_word_length

        self.rnn = nn.LSTM(input_size=rnn_input_size,
                           hidden_size=self.hidden_dim,
                           num_layers=self.num_layers,
                           bidirectional=True,
                           batch_first=True)

        self.miss_linear = nn.Linear(
            self.vocab_size, config.get('miss_linear_dim', 50))

        # The input dimension for the linear layer should be the LSTM output dimension (hidden_dim * 2)
        # plus the dimension of miss_chars if concatenated.
        linear_input_dim = self.hidden_dim * \
            2 + config.get('miss_linear_dim', 50)

        self.linear_out = nn.Linear(linear_input_dim, self.vocab_size)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        # Move all layers to the correct device
        self.to(self.device)

    def forward(self, x, x_lens, miss_chars):
        # Move input data to the correct device
        x, x_lens, miss_chars = x.to(
            self.device), x_lens, miss_chars.to(self.device)

        # print("Shape of input x:", x.shape)
        # print("Shape of input miss_chars:", miss_chars.shape)

        # Reshaping x to separate features and character indices
        batch_size, max_seq_length, feature_size = x.shape
        num_features_per_word = feature_size // self.max_word_length
        reshaped_x = x.view(batch_size, max_seq_length,
                            self.max_word_length, -1)

        char_indices = reshaped_x[:, :, :, 0].long()
        other_features = reshaped_x[:, :, :, 1:]

        # Flatten char_indices for embedding
        char_indices_flattened = char_indices.view(-1, char_indices.size(-1))

        # Reshape other_features for concatenation
        other_features_reshaped = other_features.reshape(
            batch_size * max_seq_length, self.max_word_length, -1)

        # Embedding and concatenation
        if self.use_embedding:
            embedded_chars = self.embedding(char_indices_flattened)
            # Concatenation
            rnn_input = torch.cat(
                (embedded_chars, other_features_reshaped), dim=-1)

            # print(f'rnn input shape: ', rnn_input.shape)
        else:
            rnn_input = other_features_reshaped

        # Make sure rnn_input is reshaped back to (batch_size, seq_length, features)
        rnn_input = rnn_input.view(batch_size, max_seq_length, -1)
        # print("Reshaped rnn input:", rnn_input.shape)

        # Packing the sequence
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(
            rnn_input, x_lens, batch_first=True, enforce_sorted=False)

        # print("Shape of packed x:", x_packed.data.shape)

        # LSTM processing
        output_packed, (hidden, cell) = self.rnn(x_packed)

        # Unpack the sequence
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            output_packed, batch_first=True)

        # Process missed characters - ensure it remains 3D even for batch size of 1
        miss_chars_processed = self.miss_linear(miss_chars)
        if miss_chars_processed.dim() == 2:
            miss_chars_processed = miss_chars_processed.unsqueeze(0)

        # Combine hidden states for bidirectional RNN
        if self.rnn.bidirectional:
            hidden_forward = hidden[-2, :, :]
            hidden_backward = hidden[-1, :, :]
            hidden_combined = torch.cat(
                (hidden_forward, hidden_backward), dim=1)
        else:
            hidden_combined = hidden[-1]

        # # Process missed characters
        # miss_chars_processed = self.miss_linear(miss_chars.squeeze())

        # Debug prints
        # print("Shape of hidden_combined before unsqueeze:", hidden_combined.shape)
        # print("Shape of miss_chars_processed before unsqueeze:",
            #   miss_chars_processed.shape)

        # Correctly reshape hidden_combined to match the batch and sequence length of miss_chars_processed
        hidden_combined = hidden_combined.unsqueeze(
            1).expand(-1, miss_chars_processed.size(1), -1)

        # Debug prints
        # print("Shape of hidden_combined after unsqueeze and expand:",
            #   hidden_combined.shape)
        # print("Shape of reshaped miss_chars_processed:",
            #   miss_chars_processed.shape)

        # Concatenate along the last dimension
        concatenated = torch.cat(
            (hidden_combined, miss_chars_processed), dim=2)

        out = self.linear_out(F.relu(concatenated))

        # print(f'out shape: ', out.shape)

        # Reshape to the desired output
        return out

    def calculate_loss(self, model_out, labels,
                       input_lens, miss_chars, vocab_size):
        # print(f'model out: ', model_out.shape)
        # print(f'labels: ', labels.shape)

        model_out = model_out.to(self.device)
        # Ensure labels are in float for BCEWithLogitsLoss
        labels = labels.to(self.device).float()
        miss_chars = miss_chars.to(self.device)
        input_lens = input_lens.to(self.device)

        # Apply sigmoid to model output (BCEWithLogitsLoss does this internally, so this is for miss_penalty calculation)
        outputs = torch.sigmoid(model_out)

        # print(f'output shape: ', outputs.shape)

        # Since miss_chars already has the same shape as outputs, we don't need to expand or unsqueeze
        # Directly calculate miss_penalty
        miss_penalty = torch.sum(outputs * miss_chars) / outputs.numel()

        # Weights per example
        weights_orig = (1 / input_lens.float()) / torch.sum(1 / input_lens)
        weights = weights_orig.unsqueeze(1).unsqueeze(
            2).expand(-1, model_out.size(1), -1).to(self.device)

        # Set up the loss function
        loss_func = nn.BCEWithLogitsLoss(weight=weights, reduction='none')

        # Calculate loss
        loss = loss_func(model_out, labels)

        # Mask loss for actual sequence length
        seq_lens_mask = torch.arange(model_out.size(1),
                                     device=self.device).expand(len(input_lens),
                                                                model_out.size(1)) < input_lens.unsqueeze(1)
        loss = loss * seq_lens_mask.unsqueeze(-1).float()
        loss = loss.sum() / seq_lens_mask.sum()

        return loss, miss_penalty
