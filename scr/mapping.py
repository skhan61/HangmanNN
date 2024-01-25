from collections.abc import MutableMapping


# Function to flatten a nested dictionary
def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

# Function to reorganize the flattened data by word length


def reorganize_by_word_length(flattened_data):
    word_length_stats = {}

    for key, value in flattened_data.items():
        parts = key.split('_')
        word_length = next((int(part)
                           for part in parts if part.isdigit()), None)

        if word_length is not None:
            # Determine the stat category by excluding the word length and initial identifier
            stat_category = '_'.join(part for part in parts if not part.isdigit(
            ) and part != 'length' and part != 'wise' and part != 'performence' and part != 'stats')

            if word_length not in word_length_stats:
                word_length_stats[word_length] = {}

            word_length_stats[word_length][stat_category] = value

    return word_length_stats


def flatten_for_logging(aggregated_metrics):
    loggable_metrics = {}
    for word_len, stats in aggregated_metrics.items():
        for stat, value in stats.items():
            flattened_key = f'{stat}_{word_len}'
            loggable_metrics[flattened_key] = value
    return loggable_metrics


# Sample nested dictionary
performence_dict = {
    'length_wise_performence_stats': {
        5: {'total_games': 100, 'wins': 60, 'total_attempts_used': 300,
            'win_rate': 0.6, 'average_attempts_used': 3.0},
        6: {'total_games': 150, 'wins': 90, 'total_attempts_used': 450,
            'win_rate': 0.6, 'average_attempts_used': 3.0}
    },
    'miss_penalty': {
        5: 0.02,
        6: 0.03
    }
}

# Flattening the nested dictionary
flattened_data = flatten_dict(performence_dict) # nested dict to flatten

# Reorganizing the data by word length
aggregated_metrics = reorganize_by_word_length(flattened_data) # for sampler/data_module
print(aggregated_metrics)
# Flattening the organized data for logging
loggable_data = flatten_for_logging(aggregated_metrics)

# Displaying the loggable data
print(loggable_data)
