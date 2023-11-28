# rnn.py
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score

from scr.base_model import BaseModel


class RNN(BaseModel):
    def __init__(self, config):
        super(RNN, self).__init__(config)
        # Store important parameters
        self.rnn_name = config['rnn']
        self.input_dim = config['vocab_size'] + 1
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.embed_dim = config['embedding_dim']
        self.output_dim = config['vocab_size']
        self.input_feature_size = config['input_feature_size']
        # Add max_word_length to the model attributes
        self.max_word_length = config['max_word_length']
        self.device = torch.device("cuda" if
                                   torch.cuda.is_available()
                                   and config['use_cuda'] else "cpu")

        # Whether to use character embeddings
        if config['use_embedding']:
            self.use_embedding = True
            self.embedding = nn.Embedding(
                self.input_dim, self.embed_dim).to(self.device)
            rnn_input_size = self.embed_dim + self.input_feature_size - 1
        else:
            self.use_embedding = False
            rnn_input_size = self.input_feature_size

        # Linear layer after RNN output
        in_features = config['miss_linear_dim'] + self.hidden_dim * 2
        mid_features = config['output_mid_features']

        self.linear1_out = nn.Linear(in_features, mid_features).to(self.device)
        self.relu = nn.ReLU().to(self.device)
        self.linear2_out = nn.Linear(
            mid_features, self.output_dim).to(self.device)

        # Linear layer after missed characters
        self.miss_linear = nn.Linear(
            config['vocab_size'], config['miss_linear_dim'])

        # Declare RNN
        if self.rnn_name == 'LSTM':
            self.rnn = nn.LSTM(input_size=rnn_input_size,
                               hidden_size=self.hidden_dim,
                               num_layers=self.num_layers,
                               dropout=config['dropout'],
                               bidirectional=True,
                               batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=self.embed_dim if self.use_embedding
                              else self.input_dim,
                              hidden_size=self.hidden_dim, num_layers=self.num_layers,
                              dropout=config['dropout'],
                              bidirectional=True, batch_first=True)

        # optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=config['lr'])
        # Device handling
        # self.device = torch.device("cuda" if \
        #     torch.cuda.is_available() and config['use_cuda'] else "cpu")
        # Ensure all layers are moved to the correct device
        self.to(self.device)

    def forward(self, x, x_lens, miss_chars):
        # Move input data to the correct device
        x, x_lens, miss_chars = x.to(
            self.device), x_lens, miss_chars.to(self.device)

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
        else:
            rnn_input = other_features_reshaped

        # Packing the sequence
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(
            rnn_input, x_lens, batch_first=True, enforce_sorted=False)

        # LSTM processing and capturing the output in output_packed
        output_packed, hidden = self.rnn(x_packed)

        # Unpack the sequence
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            output_packed, batch_first=True)

        # Process missed characters
        # Squeeze miss_chars to remove any extra dimension
        miss_chars_processed = self.miss_linear(miss_chars.squeeze())

        # Combine hidden states for bidirectional RNN
        if self.rnn.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1]

        # Forward through the linear layers
        concatenated = torch.cat((hidden, miss_chars_processed), dim=1)
        out = self.linear2_out(self.relu(self.linear1_out(concatenated)))

        # Reshape to the desired output
        return out.unsqueeze(1).repeat(1, max(x_lens), 1)

    # def forward(self, x, x_lens, miss_chars):
    #     # Move input data to the correct device
    #     x, x_lens, miss_chars = x.to(
    #         self.device), x_lens, miss_chars.to(self.device)

    #     # Reshaping x to separate features and character indices
    #     batch_size, max_seq_length, feature_size = x.shape
    #     num_features_per_word = feature_size // self.max_word_length
    #     reshaped_x = x.view(batch_size, max_seq_length,
    #                         self.max_word_length, -1)

    #     char_indices = reshaped_x[:, :, :, 0].long()
    #     other_features = reshaped_x[:, :, :, 1:]

    #     # Flatten char_indices for embedding
    #     char_indices_flattened = char_indices.view(-1, char_indices.size(-1))

    #     # Reshape other_features for concatenation
    #     other_features_reshaped = other_features.reshape(batch_size
    #                                                      * max_seq_length,
    #                                                      self.max_word_length, -1)

    #     # Embedding and concatenation
    #     if self.use_embedding:
    #         embedded_chars = self.embedding(char_indices_flattened)
    #         # Debugging print statements to check shapes
    #         print("embedded_chars shape:", embedded_chars.shape)
    #         print("other_features_reshaped shape:",
    #               other_features_reshaped.shape)

    #         # Ensure the shapes are compatible for concatenation
    #         assert embedded_chars.size(0) == other_features_reshaped.size(
    #             0), "Mismatch in batch dimensions"
    #         assert embedded_chars.size(
    #             1) == self.max_word_length, "Embedded chars do not match max_word_length"

    #         # Concatenation
    #         rnn_input = torch.cat(
    #             (embedded_chars, other_features_reshaped), dim=-1)
    #     else:
    #         rnn_input = other_features_reshaped

    #     # # Reshape back to include sequence length and batch size
    #     # rnn_input = rnn_input.view(batch_size, max_seq_length, -1)

    #     # Check the shape of rnn_input and x_lens
    #     print("rnn_input shape:", rnn_input.shape)
    #     print("x_lens:", x_lens)

    #     # Packing the sequence
    #     x_packed = torch.nn.utils.rnn.pack_padded_sequence(
    #         rnn_input, x_lens, batch_first=True, enforce_sorted=False)

    #     # Debug: Check the packed sequence
    #     print("x_packed data shape:", x_packed.data.shape)
    #     print("x_packed batch_sizes:", x_packed.batch_sizes)

    #     # Debug: Check LSTM input just before processing
    #     print("About to feed into LSTM, rnn_input shape:", rnn_input.shape)

    #     #         # Debug: Print shapes just before LSTM
    #     # print("rnn_input shape (before LSTM):", rnn_input.shape)
    #     print("LSTM configuration - input size:", self.rnn.input_size)

    #     # # LSTM processing and capturing the output in output_packed
    #     # output_packed, hidden = self.rnn(x_packed)

    #     # # Unpack the sequence
    #     # output, _ = torch.nn.utils.rnn.pad_packed_sequence(
    #     #     output_packed, batch_first=True)

    #     # LSTM processing and capturing the output in output_packed
    #     output_packed, hidden = self.rnn(x_packed)

    #     # Debug: Check the total length of sequences in x_lens
    #     total_seq_length = x_lens.sum().item()
    #     print("Total sequence length in x_lens:", total_seq_length)
    #     print("Total data points in LSTM output:", output_packed.data.shape[0])

    #     # Debug: Check output shape from LSTM
    #     print("Shape of LSTM output:", output_packed.data.shape)

    #     # Unpack the sequence
    #     output, _ = torch.nn.utils.rnn.pad_packed_sequence(output_packed, \
    #         batch_first=True)

    #     # Process missed characters
    #     miss_chars = self.miss_linear(miss_chars.squeeze(
    #         1) if miss_chars.dim() > 2 else miss_chars)

    #     # Combine hidden states for bidirectional RNN
    #     if self.rnn.bidirectional:
    #         hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
    #     else:
    #         hidden = hidden[-1]

    #     # Forward through the linear layers
    #     concatenated = torch.cat((hidden, miss_chars), dim=1)
    #     out = self.linear2_out(self.relu(self.linear1_out(concatenated)))

    #     # Reshape to the desired output
    #     return out.unsqueeze(1).repeat(1, max(x_lens), 1)

    def calculate_loss(self, model_out, labels,
                       input_lens, miss_chars, vocab_size):

        batch_size, seq_len, _ = model_out.shape
        model_out = model_out.to(self.device)
        labels = labels.to(self.device).to(torch.int64)
        miss_chars = miss_chars.to(self.device)
        input_lens = input_lens.to(self.device)

        # Convert labels to one-hot encoding
        one_hot_labels = torch.zeros(
            batch_size, seq_len, vocab_size, device=self.device)
        one_hot_labels.scatter_(2, labels.unsqueeze(2), 1)

        # Apply log softmax to model output
        outputs = nn.functional.log_softmax(model_out, dim=-1)

        # # Debug: Check shapes of outputs and one_hot_labels
        # print("Debug - outputs shape:", outputs.shape)
        # print("Debug - one_hot_labels shape:", one_hot_labels.shape)

        # Calculate miss_penalty
        miss_penalty = torch.sum(outputs * miss_chars.unsqueeze(1).expand_as(outputs)) \
            / outputs.numel()

        # Weights per example
        weights_orig = (1 / input_lens.float()) / torch.sum(1 / input_lens)
        weights = weights_orig.unsqueeze(1).unsqueeze(
            2).expand(-1, seq_len, -1).to(self.device)

        # Set up the loss function
        loss_func = nn.BCEWithLogitsLoss(weight=weights, reduction='none')

        # Calculate loss
        loss = loss_func(outputs, one_hot_labels)

        # Mask loss for actual sequence length
        seq_lens_mask = torch.arange(seq_len, device=self.device).expand(
            len(input_lens), seq_len) < input_lens.unsqueeze(1)
        loss = loss * seq_lens_mask.unsqueeze(-1).float()
        loss = loss.sum() / seq_lens_mask.sum()

        return loss, miss_penalty

    # def calculate_eval_score(self, model_out, labels, input_lens, miss_chars, vocab_size):
    #     batch_size, seq_len, _ = model_out.shape
    #     model_out = model_out.to(self.device)
    #     labels = labels.to(self.device).to(torch.int64)

    #     # Convert labels to one-hot encoding
    #     one_hot_labels = torch.zeros(batch_size, seq_len, vocab_size, device=self.device)
    #     one_hot_labels.scatter_(2, labels.unsqueeze(2), 1)

    #     # Apply log softmax to model output
    #     outputs = nn.functional.log_softmax(model_out, dim=-1)

    #     # Convert to class predictions
    #     predictions = torch.argmax(outputs, dim=-1)

    #     # Flatten the predictions and labels for computing F1 score
    #     predictions_flat = predictions.view(-1)
    #     labels_flat = labels.view(-1)

    #     # Masking to consider only valid (non-padded) parts of the batch
    #     seq_lens_mask = torch.arange(seq_len).expand(len(input_lens), seq_len) < input_lens.unsqueeze(1)
    #     valid_indices = seq_lens_mask.view(-1)

    #     # Filter out invalid indices
    #     valid_predictions = predictions_flat[valid_indices]
    #     valid_labels = labels_flat[valid_indices]

    #     # Calculate precision, recall, and F1 score
    #     precision = precision_score(valid_labels.cpu().numpy(), valid_predictions.cpu().numpy(), average='weighted')
    #     recall = recall_score(valid_labels.cpu().numpy(), valid_predictions.cpu().numpy(), average='weighted')
    #     f1 = f1_score(valid_labels.cpu().numpy(), valid_predictions.cpu().numpy(), average='weighted')

    #     return precision, recall, f1
