from transformers import BertModel, BertTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from scr.base_model import BaseModel
from pathlib import Path
from scr.feature_engineering import *

class RNNWithPretrained(BaseModel):
    def __init__(self, config):
        super(RNNWithPretrained, self).__init__(config)

        # Load BERT model and tokenizer
        bert_pretrained_path = Path("/home/sayem/Desktop/Hangman/pretrained")
        self.bert = BertModel.from_pretrained(str(bert_pretrained_path))
        self.tokenizer = BertTokenizer.from_pretrained(str(bert_pretrained_path))
        bert_output_size = self.bert.config.hidden_size
        print("BERT output size:", bert_output_size)

        # RNN Configuration
        self.configure_rnn(config, bert_output_size)
        # self.max_length = config['max_length']

        # Optimizer Setup
        self.optimizer = optim.Adam(self.parameters(), lr=config['lr'])

    def configure_rnn(self, config, bert_output_size):
        """Configure the RNN layers and related settings."""
        self.rnn_name = config['rnn']
        self.hidden_dim = config['hidden_dim']
        self.num_layers = config['num_layers']
        self.output_dim = config['vocab_size']

        rnn_input_size = self.calculate_rnn_input_size(config, bert_output_size)
        print("RNN input size:", rnn_input_size)

        # RNN layers
        if self.rnn_name == 'LSTM':
            self.rnn = nn.LSTM(input_size=rnn_input_size, hidden_size=self.hidden_dim, 
                               num_layers=self.num_layers, dropout=config['dropout'], 
                               bidirectional=True, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=rnn_input_size, hidden_size=self.hidden_dim, 
                              num_layers=self.num_layers, dropout=config['dropout'], 
                              bidirectional=True, batch_first=True)

        # Linear layers setup
        self.setup_linear_layers(config)

    def calculate_rnn_input_size(self, config, bert_output_size):
        """Calculate the input size for RNN based on configurations."""
        self.input_feature_size = config['input_feature_size']
        rnn_input_size = bert_output_size + self.input_feature_size - 1
        
        if config['use_embedding']:
            self.use_embedding = True
            self.embed_dim = config['embedding_dim']
            self.embedding = nn.Embedding(config['vocab_size'] + 1, self.embed_dim)
            rnn_input_size += self.embed_dim - 1
        else:
            self.use_embedding = False

        return rnn_input_size

    def setup_linear_layers(self, config):
        """Setup linear layers after RNN."""
        in_features = config['miss_linear_dim'] + self.hidden_dim * 2
        mid_features = config['output_mid_features']
        self.linear1_out = nn.Linear(in_features, mid_features)
        self.relu = nn.ReLU()
        self.linear2_out = nn.Linear(mid_features, self.output_dim)
        self.miss_linear = nn.Linear(config['vocab_size'], config['miss_linear_dim'])

    def forward(self, x, x_lens, miss_chars):
        bert_embeddings = []
        for i in range(x.size(0)):
            masked_word = ''.join([idx_to_char[int(idx)] for idx in x[i, :, 0]])
            bert_input = self.tokenizer(masked_word, return_tensors='pt', padding=True, truncation=True, max_length=128)  # Example fixed max_length
            bert_output = self.bert(**bert_input)
            bert_embeddings.append(bert_output.last_hidden_state[:, 0, :])
        bert_embeddings = torch.stack(bert_embeddings)


        # RNN Input preparation (assuming you have a method to prepare it)
        rnn_input = self.prepare_rnn_input(x, bert_embeddings, x_lens)

        # RNN forward pass (assuming you have a method for this)
        output_packed, hidden_states = self.rnn_forward(rnn_input, x_lens)

        # Linear layers processing (assuming you have a method for this)
        out = self.process_linear_layers(hidden_states, miss_chars)
        return out


    def prepare_rnn_input(self, x, bert_embeddings, x_lens):
        """Prepare input for RNN, concatenating BERT embeddings with other features."""
        rnn_input = torch.cat((bert_embeddings, x[:, :, 1:]), dim=2)  # Skipping the masked input feature
        if self.use_embedding:
            embedding_input = x[:, :, 0].long()
            embedded = self.embedding(embedding_input)
            rnn_input = torch.cat((embedded, rnn_input), dim=2)
        print("RNN input shape:", rnn_input.shape)
        return rnn_input

    def rnn_forward(self, rnn_input, x_lens):
        """Perform RNN forward pass."""
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(rnn_input, x_lens, batch_first=True, enforce_sorted=False)
        output_packed, (hidden, cell) = self.rnn(x_packed)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output_packed, batch_first=True)
        return output_packed, (hidden, cell)

    def process_linear_layers(self, hidden_states, miss_chars):
        """Process output through linear layers."""
        hidden, cell = hidden_states
        if self.rnn.bidirectional:
            hidden = self.combine_bidirectional_hidden(hidden)
        hidden = hidden.view(hidden.size(0), -1)
        concatenated = torch.cat((hidden, miss_chars), dim=1)
        out = self.linear2_out(self.relu(self.linear1_out(concatenated)))
        return out

    def combine_bidirectional_hidden(self, hidden):
        """Combine hidden states for bidirectional RNN."""
        hidden = hidden.view(self.num_layers, 2, hidden.size(1), self.hidden_dim)
        return torch.cat((hidden[-1, 0, :, :], hidden[-1, 1, :, :]), dim=1)






    def calculate_loss(self, model_out, labels, \
        input_lens, miss_chars, vocab_size, use_cuda=False):
        
        batch_size, seq_len, output_dim = model_out.shape

        # Convert labels to one-hot encoding
        one_hot_labels = torch.zeros(batch_size, seq_len, vocab_size)
        if use_cuda:
            one_hot_labels = one_hot_labels.cuda()
            labels = labels.cuda()

        labels = labels.to(torch.int64)  # Ensure labels are of type int64
        one_hot_labels.scatter_(2, labels.unsqueeze(2), 1)

        # Apply log softmax to model output
        outputs = nn.functional.log_softmax(model_out, dim=-1)

        # # Debug: Check shapes of outputs and one_hot_labels
        # print("Debug - outputs shape:", outputs.shape)
        # print("Debug - one_hot_labels shape:", one_hot_labels.shape)

        # Calculate model output loss for miss characters
        # miss_penalty = torch.sum(outputs * miss_chars.unsqueeze(1).expand_as(outputs), dim=(0, 1)) / outputs.shape[0]
        miss_penalty = torch.sum(outputs * miss_chars.unsqueeze(1).expand_as(outputs)) / outputs.numel()

        # Weights per example is inversely proportional to length of word
        weights_orig = (1 / input_lens.float()) / torch.sum(1 / input_lens)

        # Ensure weights_orig is on the correct device
        if use_cuda:
            weights_orig = weights_orig.cuda()

        # Modify weights tensor to be broadcastable to the shape of outputs and one_hot_labels
        # The key change here is using seq_len from outputs.shape[1] instead of a fixed size
        seq_len = outputs.shape[1]  # Get the actual sequence length from outputs
        weights = weights_orig.unsqueeze(1).unsqueeze(2).expand(-1, seq_len, -1)

        # Proceed with your loss calculation...


        # # Debug: Check shape of weights
        # print("Debug - weights shape:", weights.shape)
        # Set up the loss function
        loss_func = nn.BCEWithLogitsLoss(weight=weights, reduction='none')

        # Calculate loss
        loss = loss_func(outputs, one_hot_labels)

        # # Debug: Check shape of loss
        # print("Debug - loss shape:", loss.shape)

        # Mask loss for actual sequence length
        seq_lens_mask = torch.arange(seq_len).expand(len(input_lens), seq_len) < input_lens.unsqueeze(1)
        if use_cuda:
            seq_lens_mask = seq_lens_mask.cuda()

        loss = loss * seq_lens_mask.unsqueeze(-1).float()  # Apply mask
        # Average over actual sequence lengths
        loss = loss.sum() / seq_lens_mask.sum()

        return loss, miss_penalty
