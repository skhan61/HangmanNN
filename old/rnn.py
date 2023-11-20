# rnn.py
import torch
import torch.nn as nn
import torch.optim as optim
from scr.model_checkpoint_manager \
    import ModelCheckpointManager

class RNN(ModelCheckpointManager):
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

        # Whether to use character embeddings
        if config['use_embedding']:
            self.use_embedding = True
            self.embedding = nn.Embedding(self.input_dim, self.embed_dim)
            rnn_input_size = self.embed_dim + self.input_feature_size - 1
        else:
            self.use_embedding = False
            rnn_input_size = self.input_feature_size

        # Linear layer after RNN output
        in_features = config['miss_linear_dim'] + self.hidden_dim * 2
        mid_features = config['output_mid_features']
        
        self.linear1_out = nn.Linear(in_features, mid_features)
        self.relu = nn.ReLU()
        self.linear2_out = nn.Linear(mid_features, self.output_dim)

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


    def forward(self, x, x_lens, miss_chars):
        # print("Original x shape:", x.shape)

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

        # Packing the sequence
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)

        # RNN forward pass
        output_packed, (hidden, cell) = self.rnn(x_packed)

        # print(output_packed.shape)

        # Unpack the sequence
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True)

        # print(output.shape)
        # Process missed characters
        miss_chars = self.miss_linear(miss_chars)

        # Combine hidden states for bidirectional RNN
        if self.rnn.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)  # Concatenate the hidden states from both directions
        else:
            hidden = hidden[-1]

        concatenated = torch.cat((hidden, miss_chars), dim=1)

        # Forward through the linear layers
        out = self.linear2_out(self.relu(self.linear1_out(concatenated)))

        # Reshape to the desired output: [batch size, max sequence length, output dim]
        # This matches the max sequence length of the input
        out = out.unsqueeze(1).repeat(1, max(x_lens), 1)
        
        return out


    def calculate_loss(self, model_out, labels, input_lens, miss_chars, vocab_size, use_cuda=True):
        batch_size, seq_len, _ = model_out.shape

        # Convert labels to one-hot encoding
        one_hot_labels = torch.zeros(batch_size, seq_len, vocab_size)
        
        if use_cuda:
            one_hot_labels = one_hot_labels.cuda()
        labels = labels.to(torch.int64)  # Ensure labels are of type int64
        one_hot_labels.scatter_(2, labels.unsqueeze(2), 1)

        # Apply log softmax to model output
        outputs = nn.functional.log_softmax(model_out, dim=-1)

        # Calculate model output loss for miss characters
        miss_penalty = torch.sum(outputs * \
            miss_chars.unsqueeze(1).expand_as(outputs), dim=(0, 1)) \
            / outputs.shape[0]

        # Weights per example is inversely proportional to length of word
        weights_orig = (1 / input_lens.float()) \
            / torch.sum(1 / input_lens).unsqueeze(-1)
        weights = torch.zeros((weights_orig.shape[0], 1))    
        weights[:, 0] = weights_orig

        if use_cuda:
            weights = weights.cuda()

        # Calculate loss for each sequence element
        loss_func = nn.BCEWithLogitsLoss(weight=weights, reduction='none')
        loss = loss_func(outputs, one_hot_labels)

        # Mask loss for actual sequence length
        seq_lens_mask = torch.arange(seq_len).expand(len(input_lens), seq_len) < input_lens.unsqueeze(1)
        if use_cuda:
            seq_lens_mask = seq_lens_mask.cuda()

        loss = loss * seq_lens_mask.unsqueeze(-1).float()  # Apply mask
        # Average over actual sequence lengths
        loss = loss.sum() / seq_lens_mask.sum()

        return loss, miss_penalty
