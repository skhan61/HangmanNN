import torch
import torch.nn as nn
import torch.nn.functional as F

# class Encoder(nn.Module):

#     def __init__(self, num_embeddings,
#                  embedding_dim, num_features, dropout_rate=0.5):

#         super(Encoder, self).__init__()
#         self.embedding = nn.Embedding(num_embeddings, embedding_dim)
#         self.embedding_dropout = nn.Dropout(dropout_rate)
#         # Number of features per character, including the encoded character
#         self.num_features = num_features
#         self.embedding_dim = embedding_dim

#     def forward(self, batch_features, seq_len=None, missed_chars=None):
#         print(f"{batch_features.shape}")
#         batch_size, max_seq_length, flattened_features_size = batch_features.shape
#         max_word_length = flattened_features_size // self.num_features

#         # Unflatten the features to separate characters and their features
#         unflattened_features = batch_features.view(
#             batch_size, max_seq_length, max_word_length, self.num_features)

#         # Extract encoded characters and additional features
#         encoded_chars = unflattened_features[:, :, :, 0].long()
#         additional_features = unflattened_features[:, :, :, 1:]

#         # Embedding for characters with dropout
#         embedded_chars = self.embedding_dropout(self.embedding(encoded_chars))

#         # Combine embedded characters with additional features
#         combined_features = torch.cat(
#             (embedded_chars, additional_features), dim=-1)

#         # Flatten the combined features
#         flattened_output = combined_features.view(
#             batch_size, max_seq_length, -1)

#         # [batch_size, max_seq_length, max_word_length * embedding_dim + additional features]

#         return flattened_output

# Define the Encoder class


class Encoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, max_word_length, missed_char_dim, dropout_rate=0.5):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout_rate)
        self.embedding_dim = embedding_dim
        self.max_word_length = max_word_length
        self.missed_char_dim = missed_char_dim

    def forward(self, batch_features, seq_len=None, missed_chars=None):
        print(
            f"Batch features shape (input to Encoder): {batch_features.shape}")

        batch_size, max_seq_length, total_feature_size = batch_features.shape

        # Splitting the total features
        encoded_chars = batch_features[:, :, :self.max_word_length].long()
        print(f"Encoded characters shape: {encoded_chars.shape}")

        additional_features = batch_features[:, :,
                                             self.max_word_length:total_feature_size - self.missed_char_dim]
        print(f"Additional features shape: {additional_features.shape}")

        # Embedding for characters with dropout
        embedded_chars = self.embedding_dropout(self.embedding(encoded_chars))
        print(f"Embedded characters shape: {embedded_chars.shape}")

        # Reshape embedded_chars to match the shape of additional_features
        embedded_chars = embedded_chars.view(
            batch_size, max_seq_length, self.max_word_length * self.embedding_dim)
        print(f"Reshaped embedded characters shape: {embedded_chars.shape}")

        # Combine embedded characters with additional features
        combined_features = torch.cat(
            (embedded_chars, additional_features), dim=-1)
        print(f"Combined features shape: {combined_features.shape}")

        # Flatten the combined features
        flattened_output = combined_features.view(
            batch_size, max_seq_length, -1)
        print(f"Flattened output shape: {flattened_output.shape}")

        return flattened_output


# Test function for the Encoder


def test_encoder():
    # Test parameters
    batch_size = 10
    max_seq_length = 10
    max_word_length = 29  # This should match the max_word_length used in Encoder
    embedding_dim = 50
    num_embeddings = 28
    missed_char_dim = 28
    num_features = 5

    # Calculate combined feature size per time step
    # Note: The combined feature size should consider the structure of the input to Encoder
    embedded_char_features = max_word_length * embedding_dim
    # Excluding the embedded character
    additional_features_per_char = (num_features - 1)
    additional_char_features = additional_features_per_char * max_word_length
    combined_feature_size = embedded_char_features + \
        additional_char_features + missed_char_dim

    print(f"Calculated combined feature size: {combined_feature_size}")

    # Instantiate the Encoder with max_word_length
    encoder = Encoder(num_embeddings, embedding_dim,
                      max_word_length, missed_char_dim)

    # Generate dummy input data with the dynamically calculated feature size
    features = torch.rand(batch_size, max_seq_length, combined_feature_size)

    # Test the Encoder with the dummy data
    flattened_output = encoder(features)
    print(f"flattened_output shape: {flattened_output.shape}")


if __name__ == "__main__":
    test_encoder()
