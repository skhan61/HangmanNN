import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, num_embeddings,
                 embedding_dim, num_features, dropout_rate=0.5):

        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout_rate)
        # Number of features per character, including the encoded character
        self.num_features = num_features
        self.embedding_dim = embedding_dim

    def forward(self, batch_features, seq_len=None, missed_chars=None):
        batch_size, max_seq_length, flattened_features_size = batch_features.shape
        max_word_length = flattened_features_size // self.num_features

        # Unflatten the features to separate characters and their features
        unflattened_features = batch_features.view(
            batch_size, max_seq_length, max_word_length, self.num_features)

        # Extract encoded characters and additional features
        encoded_chars = unflattened_features[:, :, :, 0].long()
        additional_features = unflattened_features[:, :, :, 1:]

        # Embedding for characters with dropout
        embedded_chars = self.embedding_dropout(self.embedding(encoded_chars))

        # Combine embedded characters with additional features
        combined_features = torch.cat(
            (embedded_chars, additional_features), dim=-1)

        # Flatten the combined features
        flattened_output = combined_features.view(
            batch_size, max_seq_length, -1)

        # [batch_size, max_seq_length, max_word_length * embedding_dim + additional features]

        return flattened_output

# Testing the Encoder with integrated feature processing


def test_encoder():
    # Test parameters
    batch_size = 10
    max_seq_length = 10
    max_word_length = 29
    num_embeddings = 28
    embedding_dim = 50
    num_features = 5

    # Instantiate the Encoder
    encoder = Encoder(num_embeddings, embedding_dim, num_features)

    # Generate dummy input data
    features = torch.rand(batch_size, max_seq_length,
                          max_word_length * num_features)

    # # Adjust original_seq_lens and missed_chars to match the new dimensions
    # original_seq_lens = torch.randint(1, max_seq_length + 1, (batch_size,))
    # missed_chars = torch.rand(batch_size, max_seq_length)

    # Test the Encoder with the dummy data
    flattened_output = encoder(features)
    print(f"flattened_output shape: {flattened_output.shape}")


if __name__ == "__main__":
    test_encoder()
