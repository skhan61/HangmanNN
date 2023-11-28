def collate_fn(batch):
    batch_features, batch_labels, \
        batch_missed_chars, original_words = zip(*batch)

    max_length = max(feature.size(0) for feature in batch_features)

    padded_features = []
    padded_labels = []

    for feature, label in zip(batch_features, batch_labels):
        # print(f"Feature tensor shape before padding: {feature.shape}")

        # Check if feature tensor has the expected number of dimensions
        if feature.dim() == 2 and feature.size(1) > 0:
            padded_feature = F.pad(feature, (0, 0, 0,
                                             max_length - feature.size(0)))
            # print('okay')
        else:
            # Handle tensors that do not have the expected shape
            # Example: Skip this feature or apply a different padding logic
            # print(feature)
            continue

        padded_label = F.pad(label, (0, max_length - label.size(0)))

        padded_features.append(padded_feature)
        padded_labels.append(padded_label)

    # Convert list of tensors to tensors with an added batch dimension
    padded_features = torch.stack(padded_features, dim=0)
    padded_labels = torch.stack(padded_labels, dim=0)
    batch_missed_chars = torch.stack(batch_missed_chars, dim=0)

    lengths_features = torch.tensor(
        [feature.size(0) for feature in batch_features], dtype=torch.long)

    return padded_features, padded_labels, batch_missed_chars, lengths_features, original_words


def collate_fn(batch):
    batch_features, batch_labels, \
        batch_missed_chars, original_words = zip(*batch)

    max_length = max(feature.size(0) for feature in batch_features)

    padded_features = []
    padded_labels = []

    for feature, label in zip(batch_features, batch_labels):
        # print(f"Feature tensor shape before padding: {feature.shape}")

        # Check if feature tensor has the expected number of dimensions
        if feature.dim() == 2 and feature.size(1) > 0:
            padded_feature = F.pad(feature, (0, 0, 0,
                                             max_length - feature.size(0)))
            # print('okay')
        else:
            # Handle tensors that do not have the expected shape
            # Example: Skip this feature or apply a different padding logic
            # print(feature)
            continue

        padded_label = F.pad(label, (0, max_length - label.size(0)))

        padded_features.append(padded_feature)
        padded_labels.append(padded_label)

    # Convert list of tensors to tensors with an added batch dimension
    padded_features = torch.stack(padded_features, dim=0)
    padded_labels = torch.stack(padded_labels, dim=0)
    batch_missed_chars = torch.stack(batch_missed_chars, dim=0)

    lengths_features = torch.tensor(
        [feature.size(0) for feature in batch_features], dtype=torch.long)

    return padded_features, padded_labels, batch_missed_chars, lengths_features, original_words
