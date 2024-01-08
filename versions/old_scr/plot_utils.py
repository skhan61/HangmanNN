from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


def plot_word_stats(word_length_stats, save_path):
    # Validate input format
    if not isinstance(word_length_stats, dict) or not all(isinstance(value, dict) for value in word_length_stats.values()):
        raise ValueError(
            "word_length_stats must be a dictionary of dictionaries")

    # Prepare data for plotting
    word_lengths = sorted(word_length_stats.keys())
    wins = [word_length_stats[length]['wins'] for length in word_lengths]
    losses = [word_length_stats[length]['losses'] for length in word_lengths]
    total_attempts = [word_length_stats[length]['total_attempts']
                      for length in word_lengths]
    avg_attempts = [word_length_stats[length]['total_attempts'] /
                    word_length_stats[length]['games'] for length in word_lengths]

    # Calculate Win/Loss Ratios as Percentages
    win_percentages = [(wins[i] / total_attempts[i] * 100) if total_attempts[i] != 0 else 0
                       for i in range(len(word_lengths))]
    loss_percentages = [(losses[i] / total_attempts[i] * 100) if total_attempts[i] != 0 else 0
                        for i in range(len(word_lengths))]
    # Creating subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))

    # Width of a bar
    bar_width = 0.35
    # Positions of bars on the X-axis
    r1 = range(len(word_lengths))
    r2 = [x + bar_width for x in r1]

    # Plot Wins and Losses
    axs[0, 0].bar(r1, wins, color='green', width=bar_width, label='Wins')
    axs[0, 0].bar(r2, losses, color='red', width=bar_width, label='Losses')
    axs[0, 0].set_title('Wins and Losses per Word Length')
    axs[0, 0].set_xlabel('Word Length')
    axs[0, 0].set_ylabel('Count')
    axs[0, 0].set_xticks(
        [r + bar_width/2 for r in range(len(wins))], word_lengths)
    axs[0, 0].legend()

    # Plot Total Attempts
    axs[0, 1].bar(word_lengths, total_attempts, color='blue')
    axs[0, 1].set_title('Total Attempts per Word Length')
    axs[0, 1].set_xlabel('Word Length')
    axs[0, 1].set_ylabel('Total Attempts')

    # Plot Average Attempts
    axs[1, 0].bar(word_lengths, avg_attempts, color='purple')
    axs[1, 0].set_title('Average Attempts per Word Length')
    axs[1, 0].set_xlabel('Word Length')
    axs[1, 0].set_ylabel('Average Attempts')

    # Plot Win/Loss Percentages
    axs[1, 1].bar(r1, win_percentages, color='green',
                  width=bar_width, label='Win %')
    axs[1, 1].bar(r2, loss_percentages, color='red',
                  width=bar_width, label='Loss %')
    axs[1, 1].set_title('Win and Loss Percentages per Word Length')
    axs[1, 1].set_xlabel('Word Length')
    axs[1, 1].set_ylabel('Percentage (%)')
    axs[1, 1].set_xticks(
        [r + bar_width/2 for r in range(len(win_percentages))], word_lengths)
    axs[1, 1].legend()
    axs[1, 1].yaxis.set_major_formatter(PercentFormatter())

    # Adjusting layout and saving
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def analyze_hangman_sample_practicality(original_word_list, stratified_sampling_method, num_samples):
    # Generate a stratified sample
    stratified_sample = stratified_sampling_method(
        original_word_list, num_samples)

    # Analyze and plot word length distributions
    plot_stratified_sampling_analysis(original_word_list, stratified_sample)

    # Get word length distributions
    original_lengths, sampled_lengths = get_length_distributions(
        original_word_list, stratified_sample)

    # Evaluate representation of word lengths
    representation_evaluation = {length: sampled_lengths.get(length, 0) / original_lengths.get(length, 1)
                                 for length in original_lengths}

    # Plotting the quality of sampling
    plt.figure(figsize=(10, 5))
    plt.bar(representation_evaluation.keys(),
            representation_evaluation.values(), color='purple', alpha=0.7)
    plt.xlabel('Word Length')
    plt.ylabel('Representation Ratio')
    plt.title('Quality of Sampling Analysis')
    plt.axhline(y=1, color='r', linestyle='--')
    plt.show()

    # Check for unique word inclusion
    unique_inclusion = len(set(stratified_sample)) == len(
        set(original_word_list))

    return representation_evaluation, unique_inclusion


def plot_stratified_sampling_analysis(original_word_list, stratified_sampled_word_list):
    original_lengths = Counter([len(word) for word in original_word_list])
    sampled_lengths = Counter([len(word)
                              for word in stratified_sampled_word_list])

    unique_lengths = sorted(set(original_lengths.keys())
                            | set(sampled_lengths.keys()))
    original_counts = [original_lengths[length] for length in unique_lengths]
    sampled_counts = [sampled_lengths[length] for length in unique_lengths]

    original_counts_normalized = np.array(
        original_counts) / sum(original_counts)
    sampled_counts_normalized = np.array(sampled_counts) / sum(sampled_counts)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(unique_lengths, original_counts_normalized, color='blue', alpha=0.7)
    plt.title('Normalized Original Word Length Distribution')
    plt.xlabel('Word Length')
    plt.ylabel('Normalized Frequency')

    plt.subplot(1, 2, 2)
    plt.bar(unique_lengths, sampled_counts_normalized, color='green', alpha=0.7)
    plt.title('Normalized Stratified Sample Word Length Distribution')
    plt.xlabel('Word Length')
    plt.ylabel('Normalized Frequency')
    plt.tight_layout()
    plt.show()


def get_length_distributions(original_word_list, stratified_sampled_word_list):
    original_lengths = Counter([len(word) for word in original_word_list])
    sampled_lengths = Counter([len(word)
                              for word in stratified_sampled_word_list])
    return original_lengths, sampled_lengths
