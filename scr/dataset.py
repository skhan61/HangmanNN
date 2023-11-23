import gc
import pickle
import random
from collections import defaultdict
from multiprocessing import Pool

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)
from torch.utils.data import DataLoader, Dataset

from scr.feature_engineering import (calculate_char_frequencies, char_to_idx,
                                     encode_guess, idx_to_char,
                                     process_single_word_inference)

gc.collect()


class ProcessedHangmanDataset(Dataset):
    def __init__(self, pkls_dir, char_freq_dict,
                 max_word_length, mode='individual'):
        self.char_freq_dict = char_freq_dict
        self.max_word_length = max_word_length
        self.data = []
        self.mode = mode
        self.initial_states = []
        self.final_states = []
        self.words = []

        for batch_dir in sorted(pkls_dir.iterdir(), key=lambda x: int(x.name)
                                if x.name.isdigit() else float('inf')):
            if batch_dir.is_dir():
                for pkl_file in batch_dir.glob("*.pkl"):
                    with open(pkl_file, 'rb') as file:
                        game_states = pickle.load(file)

                        # Extract word from filename
                        word = pkl_file.stem.split('_from_')[0]

                        for game_state in game_states:
                            game_won, guesses = game_state

                            if self.mode == 'individual':
                                for i in range(len(guesses) - 1):
                                    current_state, next_guess = guesses[i][1], guesses[i + 1][0]
                                    encoded_state, missed_chars = self.process_state(
                                        current_state)
                                    encoded_next_guess = self.encode_guess(
                                        next_guess)
                                    self.data.append(
                                        (encoded_state, missed_chars, encoded_next_guess))
                                    self.words.append(word)

                            elif self.mode == 'sequence':
                                if len(guesses) > 0:
                                    states = [guesses[0][1]]
                                    next_guesses = []
                                    for i in range(1, len(guesses)):
                                        next_guesses.append(guesses[i][0])
                                        states.append(guesses[i][1])
                                    self.data.append(
                                        (states[:-1], next_guesses))
                                    self.words.append(word)

                            initial_state = guesses[0][1] if guesses else None
                            final_state = guesses[-1][1] if guesses else None
                            self.initial_states.append(initial_state)
                            self.final_states.append(final_state)

    # def process_state(self, state):
    #     feature_set, missed_chars = process_single_word_inference(
    #         state, self.char_freq_dict, self.max_word_length)
    #     return feature_set, missed_chars

    # def process_state(self, state):
    #     feature_set, missed_chars = process_single_word_inference(
    #         state, self.char_freq_dict, self.max_word_length)

    #     # print(feature_set.shape)

    #     # Convert the feature set to a tensor if it's not already
    #     feature_set_tensor = torch.tensor(feature_set, dtype=torch.float32) \
    #         if not isinstance(
    #         feature_set, torch.Tensor) else feature_set

    #     # # Padding the feature set to the maximum word length
    #     # if feature_set_tensor.size(0) < self.max_word_length:
    #     #     padding_size = self.max_word_length - feature_set_tensor.size(0)
    #     #     # Pad at the end of the first dimension (rows)
    #     #     feature_set_tensor = F.pad(
    #     #         feature_set_tensor, (0, 0, 0, padding_size))

    #     # print(feature_set_tensor.shape)

    #     return feature_set_tensor, missed_chars

    def process_state(self, state):
        feature_set, missed_chars = process_single_word_inference(
            state, self.char_freq_dict, self.max_word_length)

        # Convert the feature set to a tensor if it's not already
        feature_set_tensor = torch.tensor(feature_set, dtype=torch.float32) \
            if not isinstance(feature_set, torch.Tensor) else feature_set

        # No padding to the maximum word length
        return feature_set_tensor, missed_chars


    def encode_guess(self, guess):
        return char_to_idx.get(guess, char_to_idx['_'])

    def __getitem__(self, index):
        
        data = self.data[index]
        initial_state = self.initial_states[index]
        final_state = self.final_states[index]
        word = self.words[index]

        if self.mode == 'individual':
            # Handling for individual mode
            encoded_state, missed_chars, encoded_next_guess = data
            return encoded_state, missed_chars, encoded_next_guess, \
                initial_state, final_state, word

        elif self.mode == 'sequence':
            states, next_guesses = data

            # print(f"states: ", type(states))

            return states, next_guesses, initial_state, final_state, word

    def __len__(self):
        return len(self.data)

    def collate_fn(self, batch):
        # Determine the maximum sequence length in this batch
        max_seq_len = max(len(states) for states, _, _, _, _ in batch)

        # Initialize batch lists
        padded_feature_sets_batch = []
        padded_labels_batch = []
        seq_lengths = []

        # Process each sequence in the batch
        for states, guesses, _, _, _ in batch:
            feature_sets = []
            labels_list = []

            # Flatten each state and process each guess
            for state, guess in zip(states, guesses):
                feature_set_tensor, _ = self.process_state(state)
                flattened_state = feature_set_tensor.view(-1)  # Flatten the state
                feature_sets.append(flattened_state)
                encoded_guess = self.encode_guess(guess)
                labels_list.append(encoded_guess)

            # Record actual sequence length
            seq_lengths.append(len(feature_sets))

            # Pad the sequence if shorter than the max sequence length
            while len(feature_sets) < max_seq_len:
                dummy_padding = torch.zeros(feature_sets[0].shape)
                feature_sets.append(dummy_padding)
                labels_list.append(0)

            # Add the padded sequences to the batch lists
            padded_feature_sets_batch.append(torch.stack(feature_sets))
            padded_labels_batch.append(torch.tensor(labels_list))

        # Convert lists to tensors for batching
        padded_feature_sets_batch = torch.stack(padded_feature_sets_batch)
        padded_labels_batch = torch.stack(padded_labels_batch)
        seq_lengths = torch.tensor(seq_lengths)

        return padded_feature_sets_batch, padded_labels_batch, seq_lengths


    # def collate_fn(self, batch):

    #      # Initialize lists for initial_states, final_states, and words
    #     initial_states_batch = []
    #     final_states_batch = []
    #     words_batch = []

    #     # Determine the maximum sequence length in this batch
    #     max_seq_len = max(len(states) for states, _, _, _,_ in batch)

    #     # Initialize batch lists
    #     padded_feature_sets_batch = []
    #     missed_chars_list_batch = []  # Collecting missed characters without padding
    #     padded_labels_batch = []
    #     seq_lengths = []

    #     # Process each sequence in the batch
    #     for states, guesses, _, _, _ in batch:
    #         feature_sets = []
    #         missed_chars_list = []  # For current sequence
    #         labels_list = []

    #         # Process each state and guess in the sequence
    #         for state, guess in zip(states, guesses):
    #             feature_set_tensor, missed_chars = self.process_state(state)
    #             feature_sets.append(feature_set_tensor)
    #             missed_chars_list.append(missed_chars)  # Collect missed characters
    #             encoded_guess = self.encode_guess(guess)
    #             labels_list.append(encoded_guess)

    #         # Record actual sequence length
    #         seq_lengths.append(len(feature_sets))

    #         # Pad the sequence if shorter than the max sequence length
    #         while len(feature_sets) < max_seq_len:
    #             dummy_padding = torch.zeros_like(feature_sets[0])
    #             feature_sets.append(dummy_padding)
    #             labels_list.append(0)  # Assuming 0 is a suitable padding value for labels

    #         # Add the padded sequences and other data to the batch lists
    #         padded_feature_sets_batch.append(torch.stack(feature_sets))
    #         missed_chars_list_batch.append(missed_chars_list)  # Directly appending the list
    #         padded_labels_batch.append(torch.tensor(labels_list))

    # padded_feature_sets_batch = torch.stack(padded_feature_sets_batch)
    # batch_size, seq_len, padded_word_len, features = padded_feature_sets_batch.shape
    # padded_feature_sets_batch = padded_feature_sets_batch.view(batch_size, seq_len, -1)

    # padded_labels_batch = torch.stack(padded_labels_batch)
    # seq_lengths = torch.tensor(seq_lengths)

    # # Collect initial_states, final_states, and words from the batch
    # for _, _, initial_state, final_state, word in batch:
    #     initial_states_batch.append(initial_state)
    #     final_states_batch.append(final_state)
    #     words_batch.append(word)

    # # Return the padded and batched data
    # return padded_feature_sets_batch, missed_chars_list_batch, \
    #     padded_labels_batch, seq_lengths, final_states_batch, final_states_batch, words_batch











































    # def collate_fn(self, batch):
    #     # # Unpack the batch
    #     # feature_sets, missed_chars_list, \
    #     #     encoded_next_guesses, initial_states, final_states, words = zip(*batch)

    #     batch_of_states, batch_of_next_guesses = list(zip(*batch))

    #     # Initialize lists to store feature sets and missed characters
    #     feature_sets = []
    #     missed_chars_list = []
    #     labels_list = []

    #     feature_sets_batch = []
    #     missed_chars_list_batch = []
    #     labels_list_batch = []

    #     # Process each state and guess in the sequence
    #     for states in batch_of_states:
    #         for state in states:
    #             feature_set_tensor, missed_chars = self.process_state(state)
    #             feature_sets.append(feature_set_tensor)
    #             missed_chars_list.append(missed_chars)

    #         feature_sets_batch.append(feature_sets)
    #         missed_chars_list.append(missed_chars)

    #     for guesses in batch_of_next_guesses:
    #         for guess in guesses:
    #             encoded_guess = self.encode_guess(guess)
    #             labels_list.append(encoded_guess)
    #             # print(type(encoded_guess))
    #         labels_list_batch.append(labels_list)

    #     print(len(feature_sets_batch))
    #     # print(len(labels_list_batch))

    #     # print(feature_sets_batch[0].shape)
    #     # print(feature_sets_batch[1].shape)

    #     return padded_features, padded_guesses, feature_lengths, \
    #         initial_states, final_states, words
