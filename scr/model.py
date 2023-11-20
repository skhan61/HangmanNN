# rnn.py
import torch
import torch.nn as nn
import torch.optim as optim
from scr.base_model \
    import BaseModel

from sklearn.metrics import f1_score, precision_score, recall_score

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

        self.device = torch.device("cuda" if \
            torch.cuda.is_available() and config['use_cuda'] else "cpu")

        # Whether to use character embeddings
        if config['use_embedding']:
            self.use_embedding = True
            self.embedding = nn.Embedding(self.input_dim, self.embed_dim).to(self.device)
            rnn_input_size = self.embed_dim + self.input_feature_size - 1
        else:
            self.use_embedding = False
            rnn_input_size = self.input_feature_size

        # Linear layer after RNN output
        in_features = config['miss_linear_dim'] + self.hidden_dim * 2
        mid_features = config['output_mid_features']
        
        self.linear1_out = nn.Linear(in_features, mid_features).to(self.device)
        self.relu = nn.ReLU().to(self.device)
        self.linear2_out = nn.Linear(mid_features, self.output_dim).to(self.device)

        # Linear layer after missed characters
        self.miss_linear = nn.Linear(config['vocab_size'], config['miss_linear_dim'])

        # Declare RNN
        if self.rnn_name == 'LSTM':
            self.rnn = nn.LSTM(input_size=rnn_input_size, 
                               hidden_size=self.hidden_dim, 
                               num_layers=self.num_layers,
                               dropout=config['dropout'], 
                               bidirectional=True, 
                               batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=self.embed_dim if self.use_embedding \
                else self.input_dim, hidden_size=self.hidden_dim, num_layers=self.num_layers,
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
        # print("Original x shape:", x.shape)
        # Move input data to the correct device
        x, x_lens, miss_chars = x.to(self.device), \
            x_lens, miss_chars.to(self.device)

        if self.use_embedding:
            # Embedding is applied only to the masked input (first feature)
            embedding_input = x[:, :, 0].long()
            embedded = self.embedding(embedding_input)  # Shape: [batch, seq_len, embed_dim]

            # Other features
            other_features = x[:, :, 1:]  # Shape: [batch, seq_len, num_other_features]

            # Concatenate embedded indices with other features
            x = torch.cat((embedded, other_features), dim=2)  # Shape: [batch, seq_len, embed_dim + num_other_features]
        else:
            x = x[:, :, :self.input_dim]  # Use the input features directly if not using embedding

        # print("Shape after embedding and concatenation:", x.shape)

        # print(len(x_lens))

        # Packing the sequence
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)

        # RNN forward pass
        output_packed, (hidden, cell) = self.rnn(x_packed)

        # print(output_packed)

        # Unpack the sequence
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True)

        # print(output.shape)
        # Process missed characters
        # Process missed characters
        miss_chars = self.miss_linear(miss_chars)

        # Reshape miss_chars to ensure it's 2D (batch_size, features)
        miss_chars = miss_chars.squeeze(1) if miss_chars.dim() > 2 else miss_chars

        # # Check and print dimensions
        # print(f"Hidden shape: {hidden.shape}")
        # print(f"Miss chars shape: {miss_chars.shape}")

        # Combine hidden states for bidirectional RNN
        if self.rnn.bidirectional:
            # Reshape hidden to 2D tensor (batch_size, num_directions * hidden_dim)
            hidden = hidden.view(self.num_layers, 2, x.size(0), self.hidden_dim)
            hidden = torch.cat((hidden[-1, 0, :, :], hidden[-1, 1, :, :]), dim=1)
        else:
            hidden = hidden[-1]

        # Ensure hidden is 2D (batch_size, features)
        hidden = hidden.view(x.size(0), -1)

        concatenated = torch.cat((hidden, miss_chars), dim=1)

        # Forward through the linear layers
        out = self.linear2_out(self.relu(self.linear1_out(concatenated)))

        # Reshape to the desired output
        out = out.unsqueeze(1).repeat(1, max(x_lens), 1)
        
        return out


    def calculate_loss(self, model_out, labels, \
        input_lens, miss_chars, vocab_size):
        
        batch_size, seq_len, _ = model_out.shape
        model_out = model_out.to(self.device)
        labels = labels.to(self.device).to(torch.int64)
        miss_chars = miss_chars.to(self.device)
        input_lens = input_lens.to(self.device)

        # Convert labels to one-hot encoding
        one_hot_labels = torch.zeros(batch_size, seq_len, vocab_size, device=self.device)
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
        weights = weights_orig.unsqueeze(1).unsqueeze(2).expand(-1, seq_len, -1).to(self.device)

        # Set up the loss function
        loss_func = nn.BCEWithLogitsLoss(weight=weights, reduction='none')

        # Calculate loss
        loss = loss_func(outputs, one_hot_labels)

        # Mask loss for actual sequence length
        seq_lens_mask = torch.arange(seq_len, device=self.device).expand(len(input_lens), seq_len) < input_lens.unsqueeze(1)
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


