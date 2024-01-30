import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim,
                 max_word_length, char_feature_dim,
                 additional_state_features, dropout_rate=0.5):
        
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_dropout = nn.Dropout(dropout_rate)
        self.embedding_dim = embedding_dim
        self.max_word_length = max_word_length
        self.char_feature_dim = char_feature_dim  # Features per character
        
        # Additional features per state
        self.additional_state_features = additional_state_features

    def forward(self, batch_features, seq_len=None, missed_chars=None):
        batch_size, max_seq_length, _ = batch_features.shape
        # print(f"Input batch_features shape: {batch_features.shape}")

        # Extract character features and reshape
        char_features = batch_features[:, :, :self.max_word_length
                                    * self.char_feature_dim].view(
                                    batch_size, max_seq_length, 
                                    self.max_word_length, self.char_feature_dim)
        # print(f"Reshaped char_features shape: {char_features.shape}")

        # Flatten for embedding and convert to long
        char_features_flat = char_features.reshape(
            -1, self.char_feature_dim).long()
        # print(
        #     f"Flattened char_features for embedding shape: {char_features_flat.shape}")

        # Embedding for characters with dropout
        embedded_chars = self.embedding_dropout(
            self.embedding(char_features_flat))
        # print(f"Shape after embedding and dropout: {embedded_chars.shape}")

        # Combine the embeddings across the char_feature_dim dimension (summing in this case)
        embedded_chars_combined = embedded_chars.sum(dim=1)
        # print(
        #     f"Shape after combining character embeddings: {embedded_chars_combined.shape}")

        # Reshape combined embeddings to match the batch and sequence structure
        embedded_chars_reshaped = embedded_chars_combined.reshape(
            batch_size, max_seq_length, self.max_word_length * self.embedding_dim)
        # print(
        #     f"Reshaped embedded_chars shape: {embedded_chars_reshaped.shape}")

        # Extract additional state features
        additional_features = batch_features[:, :,
                                             self.max_word_length * self.char_feature_dim:]
        # print(
        #     f"Extracted additional_features shape: {additional_features.shape}")

        # Combine embedded characters with additional state features
        combined_features = torch.cat(
            (embedded_chars_reshaped, additional_features), dim=-1)
        # print(
        #     f"Shape of combined_features after concatenation: {combined_features.shape}")

        return combined_features


# Test function for the Encoder


def test_encoder():
    batch_size = 10
    max_seq_length = 10
    max_word_length = 29
    char_feature_dim = 5  # Features per character
    embedding_dim = 50
    num_embeddings = 28
    num_features = 154

    additional_state_features = num_features - max_word_length * char_feature_dim

    encoder = Encoder(num_embeddings, embedding_dim, max_word_length,
                      char_feature_dim, additional_state_features)

    # 145 character features (29 * 5) + 9 additional features = 154 total features
    features = torch.rand(batch_size, max_seq_length, max_word_length *
                          char_feature_dim + additional_state_features)

    # combined_features = encoder(features)
    # print(f"Combined features shape: {combined_features.shape}")

    combined_features = encoder(features)
    print(f"Combined features shape: {combined_features.shape}")

    # Expected output shape
    expected_output_shape = (batch_size, max_seq_length,
                             max_word_length * embedding_dim + additional_state_features)

    # Verify the output shape matches the expected shape
    assert combined_features.shape == expected_output_shape, f"Output shape mismatch: expected {expected_output_shape}, got {combined_features.shape}"

    print("Output shape matches the calculated shape.")


if __name__ == "__main__":
    test_encoder()
